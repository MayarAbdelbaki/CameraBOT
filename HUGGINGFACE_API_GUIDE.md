# 🤗 Hugging Face API Configuration Guide

This guide helps you fix the **405 Method Not Allowed** error when sending data to Hugging Face.

---

## 📌 The Problem

The error **"POST method not allowed. No actions exist for this page (status: 405)"** occurs when:

1. ❌ You're using the **wrong API endpoint URL**
2. ❌ You're confusing **Gradio Space** with **Inference API**
3. ❌ The URL format is incorrect

---

## ✅ Solution: Choose the Right API Type

### **Option 1: Gradio Space** (Recommended for custom apps)

If you deployed a Gradio app on Hugging Face Spaces:

#### ✅ Correct URL Format:
```
https://username-spacename.hf.space/api/predict
```

#### Example:
- **Space name**: `mayarelshamy/ssig`
- **Correct URL**: `https://mayarelshamy-ssig.hf.space/api/predict`
- **Request format**: JSON with base64 encoded image

#### Configuration:
```python
# In face_capture.py
face_capture = FaceCapture(
    save_folder="captured_faces",
    width=640,
    height=480,
    use_gradio_space=True,  # ✅ Use Gradio Space
    space_name="mayarelshamy/ssig",  # Your space name
    hf_api_token="hf_xxxxx"  # Optional for public spaces
)
```

#### Or using environment variables (.env file):
```env
USE_GRADIO_SPACE=True
SPACE_NAME=mayarelshamy/ssig
HF_API_KEY=hf_xxxxx
```

---

### **Option 2: Inference API** (For pre-trained models)

If you're using Hugging Face's Inference API with a deployed model:

#### ✅ Correct URL Format:
```
https://api-inference.huggingface.co/models/username/model-name
```

#### Example:
- **Model name**: `username/my-face-detection-model`
- **Correct URL**: `https://api-inference.huggingface.co/models/username/my-face-detection-model`
- **Request format**: Raw image bytes

#### Configuration:
```python
# In face_capture.py
face_capture = FaceCapture(
    save_folder="captured_faces",
    width=640,
    height=480,
    use_gradio_space=False,  # ✅ Use Inference API
    space_name="username/my-model",  # Your model name
    hf_api_token="hf_xxxxx"  # Required for Inference API
)
```

#### Or using environment variables (.env file):
```env
USE_GRADIO_SPACE=False
SPACE_NAME=username/my-model
HF_API_KEY=hf_xxxxx
```

---

## 🔍 How to Find Your Correct Configuration

### Step 1: Identify Your Setup

**Do you have a Gradio Space?**
- Go to: https://huggingface.co/spaces
- Look for your space (e.g., `mayarelshamy/ssig`)
- If you see a Gradio interface → Use **Option 1 (Gradio Space)**

**Do you have a deployed model?**
- Go to: https://huggingface.co/models
- Look for your model
- If it has an "Inference API" section → Use **Option 2 (Inference API)**

### Step 2: Get Your Space/Model Name

Look at the URL of your Space or Model:
- `https://huggingface.co/spaces/mayarelshamy/ssig` → Space name is `mayarelshamy/ssig`
- `https://huggingface.co/username/model-name` → Model name is `username/model-name`

### Step 3: Get Your API Token (if needed)

1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name (e.g., "CameraBOT")
4. Select permissions (read for public models, write if needed)
5. Copy the token (starts with `hf_`)

---

## 🛠️ Fixed Code Changes

Both `face_capture.py` and `camera_person_detection.py` have been updated to:

✅ **Automatically construct the correct URL** based on your configuration  
✅ **Support both Gradio Space and Inference API**  
✅ **Send data in the correct format** (JSON for Gradio, raw bytes for Inference API)  
✅ **Provide helpful error messages** when something goes wrong  

---

## 📝 Quick Setup Steps

### For `face_capture.py`:

1. **Edit the configuration** in the `main()` function (line 305-312):

```python
face_capture = FaceCapture(
    save_folder="captured_faces",
    width=640,
    height=480,
    use_gradio_space=True,  # Change based on your setup
    space_name="YOUR-USERNAME/YOUR-SPACE-NAME",  # ⚠️ CHANGE THIS
    hf_api_token="YOUR-API-TOKEN-HERE"  # ⚠️ CHANGE THIS
)
```

### For `camera_person_detection.py`:

1. **Create a `.env` file** from `.env.example`:
   ```bash
   cp .env.example .env
   ```

2. **Edit the `.env` file** with your settings:
   ```env
   USE_GRADIO_SPACE=True
   SPACE_NAME=YOUR-USERNAME/YOUR-SPACE-NAME
   HF_API_KEY=YOUR-API-TOKEN-HERE
   ```

---

## 🧪 Test Your Configuration

Run your script and look for these messages:

### ✅ Success:
```
📡 Using Gradio Space API: https://mayarelshamy-ssig.hf.space/api/predict
Camera initialized: 640x480
Face detector initialized
✓ Image sent to Hugging Face successfully
```

### ❌ Error 405:
```
✗ Failed to send image to Hugging Face. Status code: 405
💡 Error 405 (Method Not Allowed) - Possible causes:
   1. Wrong endpoint URL - check if it's a Space or Inference API
   2. For Gradio Space, use: https://username-spacename.hf.space/api/predict
   3. For Inference API, use: https://api-inference.huggingface.co/models/username/model
```

If you see error 405, double-check:
- Is `use_gradio_space` set correctly?
- Is your `space_name` correct?
- Does the Space/Model exist and is it public?

---

## 📚 Additional Resources

- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Hugging Face Inference API Documentation](https://huggingface.co/docs/api-inference/index)
- [Gradio API Documentation](https://gradio.app/sharing-your-app/#api-page)

---

## 💬 Common Issues

### Issue: "Space/Model not found (404)"
**Solution**: Check if the space/model name is correct and the space/model is public.

### Issue: "Unauthorized (401/403)"
**Solution**: Check if your API token is valid and has the right permissions.

### Issue: "Model is loading (503)"
**Solution**: The model might be cold-starting. Wait a few seconds and try again.

---

## 🎯 Summary

| API Type | URL Format | Request Format | Token Required |
|----------|------------|----------------|----------------|
| **Gradio Space** | `https://username-spacename.hf.space/api/predict` | JSON with base64 image | Optional (for public spaces) |
| **Inference API** | `https://api-inference.huggingface.co/models/username/model` | Raw image bytes | Yes |

Choose the right type based on your setup, and the code will handle the rest! 🚀

