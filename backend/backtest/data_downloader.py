"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 6
Chapter 1: High-Performance Historical Data Ingestion

File: backend/backtest/data_downloader.py
Purpose: Fetch tick-level Binance historical data with resume capability.
Constraints: Must process millions of ticks without exceeding 8GB RAM.
Features: 
    - Async downloads with aiohttp
    - Automatic resume on interruption
    - Checkpointing to avoid re-downloading
    - Strict memory management via chunked processing
"""

import asyncio
import aiohttp
import json
import os
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import struct
import logging

# Configure logging for production monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BinanceDataDownloader:
    """
    Ultra-efficient historical data downloader for Binance futures.
    Implements checkpoint-based resumption and memory-safe chunking.
    """
    
    __slots__ = [
        'base_url', 
        'output_dir', 
        'symbols', 
        'chunk_size_ms', 
        'max_retries', 
        'timeout',
        'checkpoint_file'
    ]
    
    def __init__(
        self, 
        output_dir: str = "./data/historical",
        symbols: List[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        chunk_size_ms: int = 3600000,  # 1 hour chunks
        max_retries: int = 5,
        timeout: int = 30
    ):
        self.base_url = "https://futures.binance.com/fapi/v1/klines"
        self.output_dir = Path(output_dir)
        self.symbols = symbols
        self.chunk_size_ms = chunk_size_ms
        self.max_retries = max_retries
        self.timeout = timeout
        self.checkpoint_file = self.output_dir / "download_checkpoints.json"
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_checkpoint_path(self, symbol: str) -> Path:
        """Generate checkpoint file path for a specific symbol."""
        return self.output_dir / f"{symbol}_checkpoint.json"
    
    def _load_checkpoint(self, symbol: str) -> Dict:
        """Load download progress checkpoint."""
        checkpoint_path = self._get_checkpoint_path(symbol)
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning(f"Corrupted checkpoint for {symbol}, starting fresh")
        return {"last_timestamp": None, "completed_chunks": []}
    
    def _save_checkpoint(self, symbol: str, checkpoint: Dict) -> None:
        """Save download progress checkpoint atomically."""
        checkpoint_path = self._get_checkpoint_path(symbol)
        temp_path = checkpoint_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w') as f:
                json.dump(checkpoint, f)
            temp_path.replace(checkpoint_path)
        except IOError as e:
            logger.error(f"Failed to save checkpoint for {symbol}: {e}")
    
    async def _fetch_chunk(
        self, 
        session: aiohttp.ClientSession, 
        symbol: str, 
        start_time: int,
        end_time: int
    ) -> Optional[List]:
        """
        Fetch a single time chunk from Binance API.
        Implements exponential backoff retry logic.
        """
        params = {
            'symbol': symbol,
            'interval': '1s',  # Tick-level data (1-second candles)
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1000  # Max allowed by Binance
        }
        
        for attempt in range(self.max_retries):
            try:
                async with session.get(
                    self.base_url, 
                    params=params, 
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if not data:
                            return []
                        return data
                    elif response.status == 429:  # Rate limited
                        retry_after = int(response.headers.get('Retry-After', 2 ** attempt))
                        logger.warning(f"Rate limited for {symbol}, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
                    else:
                        logger.error(f"API error for {symbol}: {response.status}")
                        return None
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching {symbol} at {start_time}, attempt {attempt + 1}")
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Unexpected error fetching {symbol}: {e}")
                await asyncio.sleep(2 ** attempt)
        
        logger.error(f"Max retries exceeded for {symbol} at {start_time}")
        return None
    
    async def _download_symbol(
        self, 
        session: aiohttp.ClientSession, 
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """
        Download all data for a single symbol with checkpointing.
        Returns total number of records downloaded.
        """
        checkpoint = self._load_checkpoint(symbol)
        last_timestamp = checkpoint.get("last_timestamp")
        completed_chunks = set(checkpoint.get("completed_chunks", []))
        
        if last_timestamp is None:
            current_start = int(start_date.timestamp() * 1000)
        else:
            current_start = last_timestamp
        
        end_timestamp = int(end_date.timestamp() * 1000)
        total_records = 0
        chunk_count = 0
        
        while current_start < end_timestamp:
            chunk_id = f"{symbol}_{current_start}"
            if chunk_id in completed_chunks:
                logger.debug(f"Skipping completed chunk {chunk_id}")
                current_start += self.chunk_size_ms
                continue
            
            chunk_end = min(current_start + self.chunk_size_ms, end_timestamp)
            
            logger.info(f"Downloading {symbol} from {datetime.fromtimestamp(current_start/1000)} to {datetime.fromtimestamp(chunk_end/1000)}")
            
            data = await self._fetch_chunk(session, symbol, current_start, chunk_end)
            
            if data is None:
                logger.error(f"Failed to download chunk {current_start}, skipping")
                current_start += self.chunk_size_ms
                continue
            
            if data:
                # Save chunk immediately to avoid memory buildup
                records_written = await self._save_chunk_to_parquet(symbol, current_start, data)
                total_records += records_written
                
                # Update checkpoint
                completed_chunks.add(chunk_id)
                checkpoint["last_timestamp"] = chunk_end
                checkpoint["completed_chunks"] = list(completed_chunks)
                self._save_checkpoint(symbol, checkpoint)
                
                chunk_count += 1
                logger.info(f"Saved {records_written} records for {symbol}, total: {total_records}")
            
            current_start = chunk_end
            
            # Memory safety: yield control to event loop
            if chunk_count % 10 == 0:
                await asyncio.sleep(0.1)
        
        return total_records
    
    async def _save_chunk_to_parquet(
        self, 
        symbol: str, 
        timestamp: int, 
        data: List
    ) -> int:
        """
        Save data chunk to parquet format efficiently.
        Uses pyarrow for zero-copy writes where possible.
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            
            # Parse Binance kline data structure
            # [open_time, open, high, low, close, volume, close_time, quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
            if not data:
                return 0
            
            # Extract columns
            open_times = [d[0] for d in data]
            opens = [float(d[1]) for d in data]
            highs = [float(d[2]) for d in data]
            lows = [float(d[3]) for d in data]
            closes = [float(d[4]) for d in data]
            volumes = [float(d[5]) for d in data]
            close_times = [d[6] for d in data]
            quote_volumes = [float(d[7]) for d in data]
            trade_counts = [int(d[8]) for d in data]
            
            table = pa.table({
                'timestamp': pa.array(open_times, type=pa.int64()),
                'open': pa.array(opens, type=pa.float64()),
                'high': pa.array(highs, type=pa.float64()),
                'low': pa.array(lows, type=pa.float64()),
                'close': pa.array(closes, type=pa.float64()),
                'volume': pa.array(volumes, type=pa.float64()),
                'close_time': pa.array(close_times, type=pa.int64()),
                'quote_volume': pa.array(quote_volumes, type=pa.float64()),
                'trade_count': pa.array(trade_counts, type=pa.int32()),
                'symbol': pa.array([symbol] * len(data), type=pa.string())
            })
            
            # Write to monthly partitioned parquet files
            date_str = datetime.fromtimestamp(timestamp / 1000).strftime('%Y_%m')
            output_file = self.output_dir / f"{symbol}_{date_str}.parquet"
            
            # Append mode if file exists
            if output_file.exists():
                existing_table = pq.read_table(output_file)
                combined_table = pa.concat_tables([existing_table, table])
                pq.write_table(combined_table, output_file, compression='snappy')
            else:
                pq.write_table(table, output_file, compression='snappy')
            
            return len(data)
            
        except ImportError:
            logger.warning("PyArrow not available, falling back to JSON storage")
            # Fallback to JSON if pyarrow not installed
            output_file = self.output_dir / f"{symbol}_{timestamp}.json"
            with open(output_file, 'w') as f:
                json.dump(data, f)
            return len(data)
        except Exception as e:
            logger.error(f"Error saving parquet file: {e}")
            return 0
    
    async def download_all(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, int]:
        """
        Download historical data for all configured symbols.
        Supports automatic resumption from checkpoints.
        """
        logger.info(f"Starting download from {start_date} to {end_date}")
        logger.info(f"Symbols: {self.symbols}")
        
        results = {}
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._download_symbol(session, symbol, start_date, end_date)
                for symbol in self.symbols
            ]
            
            # Process symbols sequentially to respect RAM limits
            for i, task in enumerate(tasks):
                symbol = self.symbols[i]
                logger.info(f"Processing symbol {i+1}/{len(self.symbols)}: {symbol}")
                record_count = await task
                results[symbol] = record_count
                logger.info(f"Completed {symbol}: {record_count} records")
                
                # Force garbage collection between symbols
                import gc
                gc.collect()
        
        logger.info(f"Download complete. Total records: {sum(results.values())}")
        return results


async def main():
    """Main entry point for data downloading."""
    downloader = BinanceDataDownloader(
        output_dir="./data/historical",
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        chunk_size_ms=3600000  # 1-hour chunks for memory safety
    )
    
    # Default: Last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Override with command line args if needed
    import sys
    if len(sys.argv) >= 3:
        start_date = datetime.strptime(sys.argv[1], '%Y-%m-%d')
        end_date = datetime.strptime(sys.argv[2], '%Y-%m-%d')
    
    results = await downloader.download_all(start_date, end_date)
    
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    for symbol, count in results.items():
        print(f"{symbol}: {count:,} records")
    print(f"Total: {sum(results.values()):,} records")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
