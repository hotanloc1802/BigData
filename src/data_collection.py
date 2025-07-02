import asyncio
import json
import os
from TikTokApi import TikTokApi

def parse_comment_dict(comment_dict, video_id=None, video_url=None): # Giữ nguyên tham số
    """
    Phân tích và trích xuất thông tin cần thiết từ dictionary bình luận thô.
    """
    try:
        user_data = comment_dict.get("user", {})
        
        # Lấy danh sách phản hồi thô, mặc định là danh sách rỗng nếu không có hoặc là None
        raw_replies = comment_dict.get("reply_comment") or [] 
        
        # Đảm bảo raw_replies là một list trước khi lặp
        parsed_replies = []
        if isinstance(raw_replies, list):
            for reply_raw_dict in raw_replies:
                # Đảm bảo từng phản hồi riêng lẻ không phải là None trước khi phân tích
                if reply_raw_dict: 
                    # Truyền video_id và video_url cho các phản hồi con
                    parsed_reply = parse_comment_dict(reply_raw_dict, video_id, video_url) 
                    if parsed_reply:
                        parsed_replies.append(parsed_reply)


        result = { # Tạo dictionary kết quả
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
            "replies": parsed_replies # Sử dụng danh sách phản hồi đã được xử lý
        }
        
        # Thêm video_id và video_url vào comment nếu được cung cấp
        if video_id:
            result["video_id"] = video_id
        if video_url:
            result["video_url"] = video_url
            
        return result
    except Exception as e:
        comment_id = comment_dict.get("cid", "Unknown ID")
        print(f"❌ Lỗi khi phân tích dictionary bình luận (ID: {comment_id}): {e}")
        return None

async def fetch_tiktok_comments(ms_token: str, video_url: str, output_filepath: str, count: int = 50):
    """
    Thu thập bình luận từ một video TikTok và lưu vào file JSON.

    Args:
        ms_token (str): ms_token hợp lệ để tạo phiên TikTokApi.
        video_url (str): URL của video TikTok cần thu thập bình luận.
        output_filepath (str): Đường dẫn đầy đủ của file JSON để lưu trữ dữ liệu.
        count (int): Số lượng bình luận chính tối đa muốn thu thập.
    """
    # Đảm bảo thư mục đầu ra tồn tại
    output_dir = os.path.dirname(output_filepath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ Đã tạo thư mục: {output_dir}")

    # Lấy video_id từ video_url
    video_id_from_url = video_url.split("/")[-1].split("?")[0] # Lấy ID từ URL

    async with TikTokApi() as api:
        try:
            # Tạo phiên sử dụng ms_token
            await api.create_sessions(
                ms_tokens=[ms_token],
                num_sessions=1,
                sleep_after=5,
                browser="chromium"
            )
            print(f"📹 Đang cố gắng lấy bình luận từ: {video_url}")
            video = api.video(url=video_url)

            comments_with_replies = []
            fetched_main_comments_count = 0

            print(f"Bắt đầu lặp qua các bình luận chính từ TikTok API cho {video_url}...")
            async for c_obj in video.comments(count=count):
                fetched_main_comments_count += 1
                main_comment_raw_data = c_obj.as_dict

                # ĐÃ SỬA: Truyền video_id và video_url khi gọi parse_comment_dict
                comment_details = parse_comment_dict(main_comment_raw_data, video_id=video_id_from_url, video_url=video_url)

                if comment_details:
                    comments_with_replies.append(comment_details)
                    print(f"  ✅ Đã xử lý bình luận chính: {comment_details.get('id')} bởi {comment_details.get('author',{}).get('username')} với {len(comment_details['replies'])} phản hồi được nhúng.")
                else:
                    print(f"❌ Bỏ qua xử lý bình luận chính thô (ID: {getattr(c_obj, 'id', 'Unknown ID')}) do lỗi trích xuất.")

            if not comments_with_replies:
                print("⚠️ Không có bình luận chính nào được lấy hoặc xử lý thành công. Điều này thường cho thấy ms_token không hợp lệ, cài đặt quyền riêng tư của video hoặc sự thay đổi đáng kể trong API của TikTok.")
                print("  Vui lòng đảm bảo ms_token của bạn là mới và hợp lệ. Cân nhắc thử một URL video khác.")
                return

            output_data = {
                "video_url": video_url,
                "video_id": video_id_from_url, # Thêm video_id vào metadata của file output
                "comments": comments_with_replies
            }

            with open(output_filepath, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            print(f"\n✅ Hoàn tất! Đã lưu {len(comments_with_replies)} bình luận chính (với các phản hồi được nhúng và tuổi dự đoán) vào {output_filepath}")

        except Exception as e:
            print(f"❌ Lỗi khi lấy video hoặc bình luận: {e}")