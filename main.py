import os
import json
import base64
import httpx
import time
import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

# Tải các biến môi trường từ file .env
load_dotenv()

# --- 1. Khởi tạo FastAPI App ---
app = FastAPI(
    title="VieDialect API",
    description="API dịch tiếng địa phương (Quảng Bình) sang tiếng phổ thông và chuyển thành giọng nói.",
    version="1.0.0"
)

# --- 2. Tích hợp Code Dịch Thuật (Gemini) ---
class GeminiDialectTranslator:
    def __init__(self, api_key: str, model_name: str = 'gemini-1.5-flash', dictionary_path: str = None):
        if not api_key:
            raise ValueError("Thiếu GEMINI_API_KEY. Hãy kiểm tra file .env của bạn.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.reference_dictionary = {}
        if dictionary_path and os.path.exists(dictionary_path):
            self.reference_dictionary = self._load_dialect_dictionary(dictionary_path)

    def _load_dialect_dictionary(self, json_file_path: str) -> dict:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _create_translation_prompt(self, dialect_text: str) -> str:
        prompt = (
            "Bạn là một chuyên gia ngôn ngữ, am hiểu sâu sắc về phương ngữ tiếng Việt, đặc biệt là tiếng Quảng Bình và tiếng phổ thông.\n"
            "Nhiệm vụ của bạn là chuyển đổi đoạn văn bản tiếng Quảng Bình sau đây sang tiếng phổ thông chuẩn Hà Nội.\n"
            "YÊU CẦU:\n"
            "1. Giữ nguyên cấu trúc câu, chỉ thay đổi nghĩa những từ địa phương.\n"
            "2. Tự nhiên, mượt mà.\n"
            "3. CHỈ trả về đoạn văn bản đã được dịch, không thêm bất kỳ lời dẫn hay giải thích nào khác.\n\n"
        )
        if self.reference_dictionary:
            prompt += "Dưới đây là một số từ địa phương Quảng Bình và nghĩa phổ thông tương đương để bạn tham khảo. Hãy ưu tiên sử dụng các nghĩa này khi gặp các từ đó, nhưng vẫn phải đảm bảo tính tự nhiên và phù hợp với ngữ cảnh của cả câu:\n"
            for key, value in self.reference_dictionary.items():
                prompt += f"- {key}: {value}\n"
            prompt += "\n"
        prompt += (
            "Dưới đây là một vài ví dụ về cách chuyển đổi:\n"
            "Ví dụ 1:\nTiếng Quảng Bình: Mi đi răng mô rứa hè?\\nTiếng phổ thông chuẩn Hà Nội: Bạn đi đâu đấy?\n\n"
            "Ví dụ 2:\nTiếng Quảng Bình: Chộ con cá ni to ghê!\\nTiếng phổ thông chuẩn Hà Nội: Nhìn thấy con cá này to thật đấy!\n\n"
            "Ví dụ 3:\nTiếng Quảng Bình: Bữa ni tau mệt trong người, khôn mần chi được.\\nTiếng phổ thông chuẩn Hà Nội: Hôm nay tôi mệt trong người, không làm gì được.\n\n"
            f"Bây giờ, hãy chuyển đổi đoạn văn bản tiếng Quảng Bình sau đây:\n\"\"\"{dialect_text}\"\"\"\n"
        )
        return prompt

    # THAY THẾ HÀM CŨ BẰNG HÀM NÀY
    def translate(self, original_text: str) -> str | None:
        """
        Dịch một chuỗi văn bản địa phương sang tiếng phổ thông.
        """
        if not original_text:
            return None
            
        detailed_prompt = self._create_translation_prompt(original_text)
        try:
            response = self.model.generate_content(detailed_prompt)
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text.strip()
            # Trả về thông báo lỗi nếu không có kết quả
            return "Không thể dịch được văn bản này."
        except Exception as e:
            print(f"Lỗi khi gọi Gemini API: {e}")
            # Ném lỗi ra để FastAPI có thể bắt và hiển thị
            raise HTTPException(status_code=500, detail=f"Lỗi từ API dịch thuật: {e}")

# --- 3. Tích hợp Code Text-to-Speech (Zalo) ---
# THAY THẾ TOÀN BỘ HÀM CŨ BẰNG HÀM NÀY
async def zalo_tts(text: str, max_retries: int = 2, delay: float = 0.5) -> bytes:
    """
    Chuyển văn bản thành giọng nói bằng Zalo TTS, có cơ chế tự động thử lại.
    """
    api_key = os.getenv("ZALO_API_KEY")
    if not api_key:
        raise ValueError("Thiếu ZALO_API_KEY. Hãy kiểm tra file .env của bạn.")

    url = "https://api.zalo.ai/v1/tts/synthesize"
    headers = {"apikey": api_key, "Content-Type": "application/x-www-form-urlencoded"}
    data = {"input": text, "speaker_id": "1", "speed": "1.0"}
    
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                # Bước 1: Gửi yêu cầu để lấy link URL của file audio
                response = await client.post(url, headers=headers, data=data)
                response.raise_for_status()
                result = response.json()
                if result.get("error_code") != 0:
                    # Nếu Zalo trả về lỗi nghiệp vụ, không cần thử lại, báo lỗi ngay
                    raise HTTPException(status_code=400, detail=f"Zalo TTS API Error: {result.get('error_message')}")
                
                audio_url = result["data"]["url"]

                # Bước 2: Dùng link vừa nhận được để tải file audio về
                audio_response = await client.get(audio_url)
                audio_response.raise_for_status() # Đây là nơi có thể xảy ra lỗi 404
                
                # Nếu đến được đây, tức là đã thành công!
                print(f"Tải audio thành công ở lần thử thứ {attempt + 1}")
                return audio_response.content

        except httpx.HTTPStatusError as e:
            last_exception = e
            if e.response.status_code == 404 and attempt < max_retries:
                # Nếu lỗi 404 và vẫn còn lượt thử, in ra thông báo và thử lại
                print(f"Lần thử {attempt + 1}: Lỗi 404 khi tải audio, thử lại sau {delay}s...")
                await asyncio.sleep(delay)
            else:
                # Nếu là lỗi HTTP khác hoặc đã hết lượt thử, ném lỗi ra ngoài
                raise HTTPException(status_code=500, detail=f"Lỗi HTTP từ Zalo: {e}")
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                print(f"Lần thử {attempt + 1}: Lỗi không xác định, thử lại sau {delay}s... ({e})")
                await asyncio.sleep(delay)
    
    # Nếu vòng lặp kết thúc mà không thành công, ném lỗi cuối cùng ra
    raise HTTPException(status_code=500, detail=f"Không thể tạo âm thanh từ Zalo sau {max_retries + 1} lần thử: {last_exception}")

# --- 4. Khởi tạo các đối tượng và định nghĩa model cho API ---
# Khởi tạo translator
translator = GeminiDialectTranslator(
    api_key=os.getenv("GEMINI_API_KEY"),
    dictionary_path="dictionary.json"
)

# Pydantic model để validate dữ liệu đầu vào
class TranslateRequest(BaseModel):
    text: str

# Pydantic model để định nghĩa dữ liệu đầu ra
class TranslateResponse(BaseModel):
    original_text: str
    translated_text: str
    audio_base64: str | None = None # Audio sẽ được trả về dưới dạng chuỗi base64

# --- 5. Xây dựng các API Endpoints ---

# Endpoint chính để xử lý dịch và TTS
@app.post("/api/translate-and-speak", response_model=TranslateResponse)
async def translate_and_speak(request: TranslateRequest):
    original_text = request.text
    if not original_text.strip():
        raise HTTPException(status_code=400, detail="Văn bản đầu vào không được để trống.")

    # Bước 1: Dịch văn bản
    translated_text = translator.translate(original_text)
    if not translated_text:
        raise HTTPException(status_code=500, detail="Không nhận được kết quả dịch.")

    # Bước 2: Chuyển văn bản đã dịch thành giọng nói
    audio_content = await zalo_tts(translated_text)
    
    # Bước 3: Mã hóa audio sang Base64 để gửi qua JSON
    audio_base64 = base64.b64encode(audio_content).decode('utf-8')

    return TranslateResponse(
        original_text=original_text,
        translated_text=translated_text,
        audio_base64=audio_base64
    )

# --- 6. Phục vụ file Frontend (Giao diện người dùng) ---
# Mount thư mục static để phục vụ file HTML, CSS, JS
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Trả về file index.html khi người dùng truy cập vào trang chủ
    return templates.TemplateResponse("index.html", {"request": request})

# --- 7. Lệnh để chạy Server (khi chạy local) ---
if __name__ == "__main__":
    import uvicorn
    # Để chạy, mở terminal và gõ: uvicorn main:app --reload
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)