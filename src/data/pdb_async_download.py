import argparse
import aiohttp
import asyncio
import os
from pathlib import Path
from tqdm.asyncio import tqdm


async def fetch(session, pdb_id, out_dir, format):
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.{format}.gz"
    out_path = out_dir / f"{pdb_id.upper()}.{format}.gz"

    if out_path.exists():
        return "exists"

    try:
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()
                out_path.write_bytes(content)
                return "downloaded"
            else:
                return f"error ({response.status})"
    except Exception as e:
        return f"failed ({str(e)})"


async def download_all(pdb_ids, out_dir, concurrency=20, format="cif"):
    out_dir.mkdir(parents=True, exist_ok=True)
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch(session, pdb_id, out_dir, format) for pdb_id in pdb_ids]
        results = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            results.append(await f)
    return results


def get_parser():
    parser = argparse.ArgumentParser(description="Download PDB files asynchronously.")
    parser.add_argument("--pdb_ids", type=str, default="pdbs.txt", help="File containing PDB IDs to download.")
    parser.add_argument("--out_dir", type=str, default="data/structures/pdb", help="Output directory for downloaded PDB files.")
    parser.add_argument("--concurrency", type=int, default=20, help="Number of concurrent downloads.")
    parser.add_argument("--format", type=str, default="cif", choices=["pdb", "cif"], help="Format of the downloaded files.")
    return parser


def parse_pdb_ids(pdb_ids_file):
    with open(pdb_ids_file, "r") as f:
        pdb_ids = [line.strip() for line in f if line.strip()]
    return pdb_ids


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    pdb_ids = parse_pdb_ids(args.pdb_ids)
    out_dir = Path(args.out_dir)
    format = args.format
    concurrency = args.concurrency

    print(f"Downloading {len(pdb_ids)} PDB files to {out_dir} with concurrency {args.concurrency}.")
    asyncio.run(download_all(pdb_ids, out_dir, concurrency, format))
