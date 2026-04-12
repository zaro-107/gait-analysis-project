import os
import yt_dlp
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------- CONFIG ---------------- #
BASE_DIR = "GaitDataset"
VIDEOS_PER_CLASS = 20
MAX_THREADS = 4   # increase if you have good internet

gait_types = {
    "normal_gait": "normal human walking gait",
    "antalgic_gait": "antalgic gait walking",
    "parkinsonian_gait": "parkinsonian gait walking",
    "spastic_gait": "spastic gait walking",
    "ataxic_gait": "ataxic gait walking",
    "steppage_gait": "steppage gait walking",
    "waddling_gait": "waddling gait walking",
    "hemiplegic_gait": "hemiplegic gait walking",
    "diplegic_gait": "diplegic gait walking",
    "choreiform_gait": "choreiform gait walking",
    "scissors_gait": "scissors gait walking",
    "neuropathic_gait": "neuropathic gait walking",
    "myopathic_gait": "myopathic gait walking",
    "cautious_gait": "cautious gait walking",
    "vestibular_gait": "vestibular gait walking",
    "sensory_ataxia": "sensory ataxia gait",
    "propulsive_gait": "propulsive gait walking",
    "magnetic_gait": "magnetic gait walking",
    "dystonic_gait": "dystonic gait walking",
    "functional_gait": "functional gait disorder walking"
}

os.makedirs(BASE_DIR, exist_ok=True)

# ---------------- DOWNLOAD FUNCTION ---------------- #
def download_single_video(entry, folder):
    video_id = entry.get("id", None)
    if not video_id:
        return

    filename = os.path.join(folder, f"{video_id}.mp4")

    # Skip if already downloaded (duplicate prevention)
    if os.path.exists(filename):
        print(f"⏭️ Skipping duplicate: {video_id}")
        return

    ydl_opts = {
        'format': 'mp4',
        'outtmpl': filename,
        'quiet': True
    }

    url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"✅ Downloaded: {video_id}")
    except Exception as e:
        print(f"❌ Failed: {video_id} | {e}")

# ---------------- FETCH + PARALLEL DOWNLOAD ---------------- #
def download_gait_class(gait, query):
    print(f"\n🚀 Processing: {gait}")

    folder = os.path.join(BASE_DIR, gait)
    os.makedirs(folder, exist_ok=True)

    search_url = f"ytsearch{VIDEOS_PER_CLASS}:{query}"

    ydl_opts = {
        'quiet': True,
        'extract_flat': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        results = ydl.extract_info(search_url, download=False)

    entries = results.get("entries", [])

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [
            executor.submit(download_single_video, entry, folder)
            for entry in entries
        ]

        for future in as_completed(futures):
            pass

    print(f"✅ Done: {gait}")

# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    for gait, query in gait_types.items():
        download_gait_class(gait, query)

    print("\n ALL DOWNLOADS COMPLETE!")
