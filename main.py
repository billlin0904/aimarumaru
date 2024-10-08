from fastapi import FastAPI, HTTPException, File, UploadFile
import os
import shutil
from auto2lrc import Auto2Lrc

app = FastAPI()

ffmpeg_dir = r'C:\ProgramData\chocolatey\bin'
os.environ["PATH"] += os.pathsep + ffmpeg_dir
os.environ["PATH"] += os.environ["PATH"] + ";" +  r"C:\ProgramData\Anaconda3\envs\aimarumaru\Lib\site-packages\nvidia\cudnn\bin" + ";" + r"C:\ProgramData\Anaconda3\envs\aimarumaru\Lib\site-packages\nvidia\cublas\bin"

auto2lrc = Auto2Lrc()

@app.get("/transcribe/")
def transcribe_audio_to_lrc(file_path: str):
    """
    接受音頻文件的路徑並返回轉錄的 LRC 文件內容字串
    """
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="音頻文件不存在")

    # 設定 LRC 文件的輸出路徑
    output_lrc_file = "output.lrc"

    try:
        # 進行轉錄並生成 LRC 文件
        auto2lrc.get_lrc(file_path, output_lrc_file)

        # 讀取 LRC 文件的內容
        with open(output_lrc_file, "r", encoding="utf-8") as f:
            lrc_content = f.read()

        # 立即刪除 LRC 文件
        os.remove(output_lrc_file)

        # 返回 LRC 文件的內容字串
        return {"lrc_content": lrc_content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"處理音頻文件時出錯: {str(e)}")


@app.post("/transcribe_file/")
async def transcribe_uploaded_audio(file: UploadFile = File(...)):
    """
    接受上傳的音頻文件，進行轉錄並返回 LRC 文件內容字串
    """
    try:
        # 保存上傳的文件到本地
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 設定 LRC 文件的輸出路徑
        output_lrc_file = f"output_{os.path.splitext(file.filename)[0]}.lrc"

        # 進行轉錄並生成 LRC 文件
        auto2lrc.get_lrc(file_location, output_lrc_file)

        # 刪除暫存的音頻文件
        os.remove(file_location)

        # 讀取 LRC 文件的內容
        with open(output_lrc_file, "r", encoding="utf-8") as f:
            lrc_content = f.read()

        # 立即刪除 LRC 文件
        os.remove(output_lrc_file)

        # 返回 LRC 文件的內容字串
        return {"lrc_content": lrc_content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"處理音頻文件時出錯: {str(e)}")
