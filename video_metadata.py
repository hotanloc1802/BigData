import asyncio
from TikTokApi import TikTokApi
import json

ms_token = "q7s50oaGgxZmEdh5Ka6SoEmAFoDb0yL-oZDabUMwXSua3FqJFtIG5NvIkg1FoXYJ9KVjGMQU8c8lqSNwF1SpnQkYO4Rt_rOgUeXOZ-U6d2VBr0ro4sRKOjZ5kFzRAc1B-82qdtbzt5R_MSpG8zsbGD1g" # Replace with your ms_token

async def main():
    async with TikTokApi() as api:
        await api.create_sessions(
            ms_tokens=[ms_token],
            num_sessions=1,
            sleep_after=3,
            browser="chromium"
        )

        # Create a list of video URLs to fetch
        video_urls_to_fetch = [
            "https://www.tiktok.com/@halinhofficial/video/7516901216467586311",
            "https://www.tiktok.com/@halinhofficial/video/7516833109401193746",
        ]

        all_fetched_videos = []

        print(f"Fetching {len(video_urls_to_fetch)} specific videos by URL...")
        for video_url in video_urls_to_fetch:
            try:
                print(f"Attempting to fetch video from URL: {video_url}...")
                video_data = await api.video(url=video_url).info() # Removed .as_dict

                # Ensure video_data is a dictionary before proceeding
                if not isinstance(video_data, dict):
                    print(f"⚠️ Warning: Expected dictionary, but got type {type(video_data)} for URL: {video_url}")
                    continue

                video_info = {
                    "id": video_data.get("id"),
                    "description": video_data.get("desc"),
                    "author": video_data.get("author", {}).get("uniqueId"),
                    "likes": video_data.get("stats", {}).get("diggCount"),
                    "shares": video_data.get("stats", {}).get("shareCount"),
                    "comments": video_data.get("stats", {}).get("commentCount"),
                    "views": video_data.get("stats", {}).get("playCount"),
                    "created_time": video_data.get("createTime")
                }
                all_fetched_videos.append(video_info)
                print(f"✅ Successfully fetched video: {video_data.get('id')} - '{video_data.get('desc', '')[:50]}...'")

            except Exception as e:
                print(f"❌ Error fetching video from URL {video_url}: {e}")

        output_filename = "video_metadata.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(all_fetched_videos, f, indent=4, ensure_ascii=False)

        print(f"\n✅ Successfully exported {output_filename} (containing information of specified videos).")

if __name__ == "__main__":
    asyncio.run(main())