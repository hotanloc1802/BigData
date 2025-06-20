import asyncio
import json
from TikTokApi import TikTokApi

ms_token = "RdfBJbfKYyJpQ-2666-YTyS7Iqm1L0cDGuM_b_yLmWjopvMogJoE0-S4q3B3Ci-2iZhqWjPFhnBa-Zgbi0IcrzSqaTg38FT9kc9iI2ONHdwX19NTNRF3SKJgKQJuq4n6Ms7ivz7yiT0Mxvf2FG6m5y_AgrE="  # 🔁 Đặt ms_token hợp lệ ở đây
video_url = "https://www.tiktok.com/@halinhofficial/video/7516833109401193746"


def parse_comment_dict(comment_dict):
    try:
        user_data = comment_dict.get("user", {})

        return {
            "id": comment_dict.get("cid"),
            "text": comment_dict.get("text"),
            "likes_count": comment_dict.get("digg_count"),
            "author": {
                "user_id": user_data.get("uid"),
                "username": user_data.get("unique_id"),
                "sec_uid": user_data.get("sec_uid"),
                "predicted_age_group": user_data.get("predicted_age_group") 
            },
            "created_time": comment_dict.get("create_time"),
        }
    except Exception as e:
        comment_id = comment_dict.get("cid", "Unknown ID")
        print(f"❌ Error parsing comment dictionary (ID: {comment_id}): {e}")
        return None

# Main script
async def main():
    async with TikTokApi() as api:
        await api.create_sessions(
            ms_tokens=[ms_token],
            num_sessions=1,
            sleep_after=5, # Tăng thời gian chờ sau khi tạo session
            browser="chromium"
        )

        print(f"📹 Attempting to fetch comments from: {video_url}")
        try:
            video = api.video(url=video_url)

            comments_with_replies = [] # Danh sách lưu trữ các bình luận và phản hồi
            fetched_main_comments_count = 0
            
            print(f"Starting to iterate through main comments from TikTok API for {video_url}...")
            async for c_obj in video.comments(count=50): 
                fetched_main_comments_count += 1
                main_comment_raw_data = c_obj.as_dict # Lấy dictionary dữ liệu thô từ đối tượng Comment
                
                comment_details = parse_comment_dict(main_comment_raw_data)
                
                if comment_details:
                    comment_details["replies"] = [] # Khởi tạo danh sách phản hồi
                    
                    # Kiểm tra và trích xuất các phản hồi được nhúng trong raw_data
                    if "reply_comment" in main_comment_raw_data and isinstance(main_comment_raw_data["reply_comment"], list):
                        for reply_raw_dict in main_comment_raw_data["reply_comment"]:
                            reply_details = parse_comment_dict(reply_raw_dict)
                            if reply_details:
                                comment_details["replies"].append(reply_details)
                                
                    comments_with_replies.append(comment_details)
                    print(f"  ✅ Processed main comment: {comment_details.get('id')} by {comment_details.get('author',{}).get('username')} with {len(comment_details['replies'])} embedded replies.")
                else:
                    print(f"❌ Skipped processing raw main comment (ID: {getattr(c_obj, 'id', 'Unknown ID')}) due to extraction error.")
            
            if not comments_with_replies:
                print("⚠️ No main comments were successfully fetched or processed. This often indicates an invalid ms_token, video privacy settings, or a significant change in TikTok's API.")
                print("   Please ensure your ms_token is fresh and valid. Consider trying a different video URL.")
                return 

            output = {
                "video_url": video_url,
                "comments": comments_with_replies
            }

            # Ghi ra file JSON
            output_filename = "video_comments.json" 
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

            print(f"\n✅ Done! Saved {len(comments_with_replies)} main comments (with embedded replies and predicted age) to {output_filename}")

        except Exception as e:
            print(f"❌ Failed to fetch video or comments: {e}")

if __name__ == "__main__":
    asyncio.run(main())