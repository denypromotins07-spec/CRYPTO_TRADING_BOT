"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 6
Chapter 1: High-Performance Historical Data Ingestion

File: backend/backtest/data_cleaner.py
Purpose: Handle missing ticks, outliers, and exchange outages in historical data.
Constraints: Must process millions of ticks without exceeding 8GB RAM.
Features:
    - Missing tick interpolation with volume-weighted methods
    - Outlier detection using MAD (Median Absolute Deviation)
    - Exchange outage detection and marking
    - Memory-efficient streaming processing
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Generator
from pathlib import Path
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataQualityFlag(Enum):
    """Flags for data quality issues."""
    CLEAN = "clean"
    MISSING_TICK = "missing_tick"
    OUTLIER = "outlier"
    EXCHANGE_OUTAGE = "exchange_outage"
    ZERO_VOLUME = "zero_volume"
    NEGATIVE_PRICE = "negative_price"
    TIMESTAMP_GAP = "timestamp_gap"


@dataclass
class CleaningStats:
    """Statistics about data cleaning operations."""
    total_rows: int = 0
    missing_ticks_filled: int = 0
    outliers_corrected: int = 0
    outage_periods_detected: int = 0
    zero_volume_rows: int = 0
    negative_price_rows: int = 0
    timestamp_gaps_fixed: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'total_rows': self.total_rows,
            'missing_ticks_filled': self.missing_ticks_filled,
            'outliers_corrected': self.missing_ticks_filled,
            'outage_periods_detected': self.outage_periods_detected,
            'zero_volume_rows': self.zero_volume_rows,
            'negative_price_rows': self.negative_price_rows,
            'timestamp_gaps_fixed': self.timestamp_gaps_fixed
        }


@dataclass
class OutlierThresholds:
    """Thresholds for outlier detection."""
    mad_multiplier: float = 5.0  # Number of MADs to consider outlier
    price_change_threshold: float = 0.10  # 10% price change threshold
    volume_spike_threshold: float = 10.0  # 10x average volume


class DataCleaner:
    """
    Production-grade data cleaner for crypto tick data.
    Implements robust statistical methods for outlier detection and correction.
    """
    
    __slots__ = [
        'symbols',
        'expected_interval_ms',
        'max_gap_ms',
        'thresholds',
        'stats',
        'outage_threshold_minutes'
    ]
    
    def __init__(
        self,
        symbols: List[str] = None,
        expected_interval_ms: int = 1000,  # 1 second
        max_gap_ms: int = 60000,  # 1 minute
        outage_threshold_minutes: int = 5,
        thresholds: OutlierThresholds = None
    ):
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        self.expected_interval_ms = expected_interval_ms
        self.max_gap_ms = max_gap_ms
        self.outage_threshold_minutes = outage_threshold_minutes
        self.thresholds = thresholds or OutlierThresholds()
        self.stats = CleaningStats()
    
    def clean_dataframe(
        self, 
        df: pd.DataFrame, 
        symbol: str = None
    ) -> Tuple[pd.DataFrame, CleaningStats]:
        """
        Clean a dataframe of tick data.
        Returns cleaned dataframe and statistics.
        """
        if df.empty:
            return df, CleaningStats()
        
        self.stats = CleaningStats()
        self.stats.total_rows = len(df)
        
        logger.info(f"Starting data cleaning for {symbol or 'unknown'}, {len(df)} rows")
        
        # Ensure timestamp column is datetime
        if 'timestamp' in df.columns:
            if df['timestamp'].dtype == 'int64':
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Step 1: Detect and fix negative prices
        df = self._fix_negative_prices(df)
        
        # Step 2: Detect and handle zero volume rows
        df = self._handle_zero_volume(df)
        
        # Step 3: Detect timestamp gaps and missing ticks
        df = self._fill_missing_ticks(df)
        
        # Step 4: Detect and correct outliers
        df = self._correct_outliers(df)
        
        # Step 5: Detect exchange outages
        df = self._mark_exchange_outages(df)
        
        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Data cleaning complete. Stats: {self.stats.to_dict()}")
        
        return df, self.stats
    
    def _fix_negative_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and fix negative prices."""
        price_cols = ['open', 'high', 'low', 'close']
        
        for col in price_cols:
            if col in df.columns:
                negative_mask = df[col] < 0
                negative_count = negative_mask.sum()
                
                if negative_count > 0:
                    self.stats.negative_price_rows += negative_count
                    logger.warning(f"Found {negative_count} negative {col} values")
                    
                    # Replace with NaN for interpolation
                    df.loc[negative_mask, col] = np.nan
        
        # Interpolate NaN values
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].interpolate(method='linear')
        
        return df
    
    def _handle_zero_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle rows with zero volume."""
        if 'volume' not in df.columns:
            return df
        
        zero_volume_mask = df['volume'] == 0
        zero_count = zero_volume_mask.sum()
        
        if zero_count > 0:
            self.stats.zero_volume_rows = zero_count
            logger.info(f"Found {zero_count} zero-volume rows")
            
            # Mark but don't remove - zero volume can be valid during low liquidity
            df.loc[zero_volume_mask, 'quality_flag'] = DataQualityFlag.ZERO_VOLUME.value
        
        return df
    
    def _fill_missing_ticks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect gaps and fill missing ticks using forward-fill with interpolation."""
        if 'timestamp' not in df.columns or len(df) < 2:
            return df
        
        # Calculate time differences
        df['_time_diff'] = df['timestamp'].diff().dt.total_seconds() * 1000
        
        # Find gaps larger than expected interval
        gap_mask = df['_time_diff'] > self.expected_interval_ms * 1.5  # 50% tolerance
        
        gap_count = gap_mask.sum()
        if gap_count > 0:
            logger.info(f"Found {gap_count} timestamp gaps")
            
            # Create complete time index
            min_ts = df['timestamp'].min()
            max_ts = df['timestamp'].max()
            
            # Only fill if gap is not too large (not an outage)
            complete_index = pd.date_range(
                start=min_ts,
                end=max_ts,
                freq=f'{self.expected_interval_ms}ms'
            )
            
            # Reindex to complete timeline
            df = df.set_index('timestamp')
            original_len = len(df)
            df = df.reindex(complete_index)
            
            # Mark newly created rows
            new_rows_mask = df.index.difference(df.dropna().index)
            df.loc[new_rows_mask, 'quality_flag'] = DataQualityFlag.MISSING_TICK.value
            
            # Forward fill then interpolate for prices
            price_cols = ['open', 'high', 'low', 'close', 'volume']
            available_cols = [c for c in price_cols if c in df.columns]
            
            for col in available_cols:
                df[col] = df[col].ffill()  # Forward fill first
                df[col] = df[col].interpolate(method='linear')  # Then interpolate
            
            filled_count = len(df) - original_len
            self.stats.missing_ticks_filled = filled_count
            logger.info(f"Filled {filled_count} missing ticks")
            
            df = df.reset_index().rename(columns={'index': 'timestamp'})
        
        # Drop temporary column
        if '_time_diff' in df.columns:
            df = df.drop('_time_diff', axis=1)
        
        return df
    
    def _correct_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and correct price outliers using MAD-based method."""
        price_cols = ['open', 'high', 'low', 'close']
        
        for col in price_cols:
            if col not in df.columns:
                continue
            
            # Calculate median and MAD
            median = df[col].median()
            mad = np.median(np.abs(df[col] - median))
            
            if mad == 0:
                continue  # No variation
            
            # Identify outliers
            z_score = np.abs(df[col] - median) / (mad * 1.4826)  # Scale factor for normal dist
            outlier_mask = z_score > self.thresholds.mad_multiplier
            
            outlier_count = outlier_mask.sum()
            if outlier_count > 0:
                self.stats.outliers_corrected += outlier_count
                logger.warning(f"Corrected {outlier_count} outliers in {col}")
                
                # Replace outliers with rolling median
                df.loc[outlier_mask, col] = df[col].rolling(
                    window=20, 
                    min_periods=1, 
                    center=True
                ).median()
        
        # Check for sudden price jumps
        if 'close' in df.columns:
            df['_price_change'] = df['close'].pct_change().abs()
            jump_mask = df['_price_change'] > self.thresholds.price_change_threshold
            
            jump_count = jump_mask.sum()
            if jump_count > 0:
                logger.warning(f"Found {jump_count} suspicious price jumps")
                df.loc[jump_mask, 'quality_flag'] = DataQualityFlag.OUTLIER.value
            
            if '_price_change' in df.columns:
                df = df.drop('_price_change', axis=1)
        
        return df
    
    def _mark_exchange_outages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and mark exchange outage periods."""
        if 'timestamp' not in df.columns or len(df) < 2:
            return df
        
        # Find large gaps that indicate outages
        df['_time_diff'] = df['timestamp'].diff().dt.total_seconds() / 60  # Convert to minutes
        
        outage_mask = df['_time_diff'] > self.outage_threshold_minutes
        outage_count = outage_mask.sum()
        
        if outage_count > 0:
            self.stats.outage_periods_detected = outage_count
            logger.warning(f"Detected {outage_count} potential exchange outage periods")
            
            df.loc[outage_mask, 'quality_flag'] = DataQualityFlag.EXCHANGE_OUTAGE.value
        
        if '_time_diff' in df.columns:
            df = df.drop('_time_diff', axis=1)
        
        return df
    
    def stream_clean_parquet_files(
        self,
        input_dir: str,
        output_dir: str,
        chunk_size: int = 100000
    ) -> Generator[CleaningStats, None, None]:
        """
        Stream-clean parquet files to avoid memory overflow.
        Yields statistics for each processed file.
        """
        try:
            import pyarrow.parquet as pq
            import pyarrow as pa
        except ImportError:
            logger.error("PyArrow not installed. Cannot process parquet files.")
            return
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        parquet_files = list(input_path.glob("*.parquet"))
        logger.info(f"Found {len(parquet_files)} parquet files to clean")
        
        for file_path in parquet_files:
            logger.info(f"Processing {file_path.name}")
            
            try:
                # Read in batches
                parquet_file = pq.ParquetFile(file_path)
                
                all_batches = []
                file_stats = CleaningStats()
                
                for batch in parquet_file.iter_batches(batch_size=chunk_size):
                    df = batch.to_pandas()
                    cleaned_df, stats = self.clean_dataframe()
                    
                    # Accumulate stats
                    for key in file_stats.__dict__:
                        file_stats.__dict__[key] += stats.__dict__.get(key, 0)
                    
                    # Convert back to arrow table
                    cleaned_table = pa.Table.from_pandas(cleaned_df)
                    all_batches.append(cleaned_table)
                
                if all_batches:
                    # Concatenate and write
                    combined_table = pa.concat_tables(all_batches)
                    output_file = output_path / f"clean_{file_path.name}"
                    pq.write_table(combined_table, output_file, compression='snappy')
                    
                    logger.info(f"Wrote cleaned data to {output_file}")
                    yield file_stats
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
    
    def validate_data_quality(
        self, 
        df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Return quality scores for the dataset.
        Used to decide if data is suitable for backtesting.
        """
        if df.empty:
            return {'overall_score': 0.0}
        
        total_rows = len(df)
        
        # Count quality flags
        flag_counts = {}
        if 'quality_flag' in df.columns:
            flag_counts = df['quality_flag'].value_counts().to_dict()
        
        # Calculate scores
        missing_ratio = flag_counts.get(DataQualityFlag.MISSING_TICK.value, 0) / total_rows
        outlier_ratio = flag_counts.get(DataQualityFlag.OUTLIER.value, 0) / total_rows
        outage_ratio = flag_counts.get(DataQualityFlag.EXCHANGE_OUTAGE.value, 0) / total_rows
        
        # Weighted scoring
        overall_score = (
            1.0 
            - (missing_ratio * 0.3) 
            - (outlier_ratio * 0.5) 
            - (outage_ratio * 0.2)
        )
        
        scores = {
            'overall_score': max(0.0, overall_score),
            'missing_ratio': missing_ratio,
            'outlier_ratio': outlier_ratio,
            'outage_ratio': outage_ratio,
            'completeness': 1.0 - missing_ratio,
            'accuracy': 1.0 - outlier_ratio
        }
        
        logger.info(f"Data quality scores: {scores}")
        
        return scores


def main():
    """Example usage of the data cleaner."""
    # Create sample data with issues
    np.random.seed(42)
    
    n_rows = 10000
    timestamps = pd.date_range('2024-01-01', periods=n_rows, freq='1s')
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': 100 + np.cumsum(np.random.randn(n_rows)),
        'high': 100 + np.cumsum(np.random.randn(n_rows)) + np.abs(np.random.randn(n_rows)),
        'low': 100 + np.cumsum(np.random.randn(n_rows)) - np.abs(np.random.randn(n_rows)),
        'close': 100 + np.cumsum(np.random.randn(n_rows)),
        'volume': np.random.exponential(1000, n_rows)
    })
    
    # Inject some issues
    # Negative prices
    df.loc[100:105, 'open'] = -50
    
    # Zero volume
    df.loc[200:210, 'volume'] = 0
    
    # Outliers
    df.loc[500, 'close'] = 500  # Huge spike
    
    # Missing ticks (remove some rows)
    df = df.drop(index=range(300, 350))
    
    # Clean the data
    cleaner = DataCleaner()
    cleaned_df, stats = cleaner.clean_dataframe(df, symbol="TEST")
    
    print("\n" + "="*60)
    print("DATA CLEANING SUMMARY")
    print("="*60)
    print(f"Total rows: {stats.total_rows:,}")
    print(f"Missing ticks filled: {stats.missing_ticks_filled:,}")
    print(f"Outliers corrected: {stats.outliers_corrected:,}")
    print(f"Zero volume rows: {stats.zero_volume_rows:,}")
    print(f"Negative price rows: {stats.negative_price_rows:,}")
    print(f"Outage periods: {stats.outage_periods_detected:,}")
    print("="*60)
    
    # Quality validation
    quality_scores = cleaner.validate_data_quality(cleaned_df)
    print(f"\nOverall Quality Score: {quality_scores['overall_score']:.2%}")
    print(f"Completeness: {quality_scores['completeness']:.2%}")
    print(f"Accuracy: {quality_scores['accuracy']:.2%}")


if __name__ == "__main__":
    main()
