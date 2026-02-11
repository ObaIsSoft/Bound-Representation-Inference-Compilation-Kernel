
import os
import asyncio
import httpx
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

async def test_apis():
    print("--- Testing yfinance ---")
    tickers = ["HG=F", "GC=F"] # Copper, Gold
    for symbol in tickers:
        try:
            print(f"Fetching {symbol}...")
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            print(f"History empty? {hist.empty}")
            if not hist.empty:
                print(f"Price: {hist['Close'].iloc[-1]}")
            else:
                print(f"No data found for {symbol}")
        except Exception as e:
            print(f"yfinance error for {symbol}: {e}")

    print("\n--- Testing MetalpriceAPI ---")
    api_key = os.getenv("METALPRICE_API_KEY")
    if not api_key:
        print("METALPRICE_API_KEY not found in env")
    else:
        print(f"Key found: {api_key[:4]}...")
        try:
            url = "https://api.metalpriceapi.com/v1/latest"
            # Try Copper (XCU) and Gold (XAU) which are often in free tiers
            params = {
                "api_key": api_key,
                "base": "USD",
                "currencies": "XCU,XAU" 
            }
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, params=params)
                print(f"Status: {resp.status_code}")
                print(f"Response: {resp.text}")
        except Exception as e:
            print(f"MetalpriceAPI error: {e}")

asyncio.run(test_apis())
