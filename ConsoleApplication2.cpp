#include <windows.h>
#include <dwmapi.h>
#include <iostream>
#include <thread>
#include <vector>
#include <string>
#include <conio.h>
#include <shellapi.h> // For SHGetKnownFolderPath
#include <ShlObj_core.h> // For FOLDERID_RoamingAppData
#include <filesystem> // For path manipulation

#include "resource.h"

#include <winrt/base.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Globalization.h>
#include <winrt/Windows.Media.Ocr.h>
#include <winrt/Windows.Graphics.Imaging.h>
#include <winrt/Windows.Storage.Streams.h>

#include <onnxruntime_cxx_api.h>
#include <sentencepiece_processor.h>

#pragma comment(lib, "Dwmapi.lib")
#pragma comment(lib, "WindowsApp.lib")
#pragma comment(lib, "Shell32.lib") // For SHGetKnownFolderPath

// WinRT namespaces
using namespace winrt;
using namespace winrt::Windows::Foundation;
using namespace winrt::Windows::Globalization;
using namespace winrt::Windows::Media::Ocr;
using namespace winrt::Windows::Graphics::Imaging;
using namespace winrt::Windows::Storage::Streams;

// ONNX Runtime and SentencePiece
namespace Ort {
    // Helper to get byte size of an ONNX tensor element type
    inline size_t GetTensorElementSize(ONNXTensorElementDataType type) {
        switch (type) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return sizeof(float);
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return sizeof(uint8_t);
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return sizeof(int8_t);
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  return sizeof(uint16_t);
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:   return sizeof(int16_t);
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return sizeof(int32_t);
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return sizeof(int64_t);
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:  return sizeof(std::string); // This is tricky, usually for string tensors, it's handled differently.
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    return sizeof(bool); // Or uint8_t
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return sizeof(uint16_t); // Half float
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  return sizeof(double);
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  return sizeof(uint32_t);
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:  return sizeof(uint64_t);
            // BFLOAT16 and other complex types might need specific handling or might not be common here.
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return sizeof(uint16_t); // Size is 2 bytes
            default:
                // Throw or return 0 to indicate an unsupported/unknown type
                // For safety, returning 0 and potentially logging an error might be better than throwing in some contexts.
                // However, if this happens, it's a fundamental issue with model compatibility.
                throw std::runtime_error("Unsupported ONNX tensor element data type in GetTensorElementSize: " + std::to_string(type));
        }
    }
} // namespace Ort

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ocr-translator-env");
Ort::SessionOptions session_options;
Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
std::unique_ptr<Ort::Session> encoder_session;
std::unique_ptr<Ort::Session> decoder_session;
sentencepiece::SentencePieceProcessor sp_source_processor;
sentencepiece::SentencePieceProcessor sp_target_processor;

// Application state
bool g_fullscreen_mode = true;
std::wstring g_current_overlay_text;
HWND g_overlay_hwnd = nullptr;
const wchar_t OVERLAY_WINDOW_CLASS[] = L"OcrTranslationOverlayWindowClass";
HINSTANCE g_hinstance = nullptr;

// Forward declarations
LRESULT CALLBACK OverlayWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
void RegisterOverlayWindowClass(HINSTANCE hInstance);
void UnregisterOverlayWindowClass(HINSTANCE hInstance);
std::wstring GetModelsDirectoryPath();
void CreateOrUpdateOverlayWindow(const std::wstring& text, const RECT& target_region);


std::wstring GetModelsDirectoryPath() {
    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    std::filesystem::path path = exePath;
    return path.parent_path() / L"models";
}

bool InitTranslationEngine()
{
    session_options.SetIntraOpNumThreads(std::thread::hardware_concurrency());
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::wstring models_dir = GetModelsDirectoryPath();
    std::wstring source_spm_path_w = models_dir + L"\\source.spm";
    std::wstring target_spm_path_w = models_dir + L"\\target.spm";
    std::wstring encoder_model_path_w = models_dir + L"\\encoder_model.onnx";
    std::wstring decoder_model_path_w = models_dir + L"\\decoder_model.onnx";

    // Helper lambda for wstring to UTF-8 string conversion
    auto wstring_to_utf8_string = [](const std::wstring& wstr) -> std::string {
        if (wstr.empty()) return std::string();
        int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.length(), NULL, 0, NULL, NULL);
        std::string strTo(size_needed, 0);
        WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.length(), &strTo[0], size_needed, NULL, NULL);
        return strTo;
    };

    // Convert wstring paths to UTF-8 string for SentencePiece
    std::string source_spm_path_s = wstring_to_utf8_string(source_spm_path_w);
    std::string target_spm_path_s = wstring_to_utf8_string(target_spm_path_w);

    const auto& source_status = sp_source_processor.Load(source_spm_path_s.c_str());
    if (!source_status.ok()) {
        std::string error_message = "Failed to load source SentencePiece model (" + source_spm_path_s + "): " + source_status.ToString();
        std::wcerr << L"[ERROR] " << std::wstring(error_message.begin(), error_message.end()) << std::endl;
        MessageBoxA(nullptr, error_message.c_str(), "Model Error", MB_OK | MB_ICONERROR);
        return false;
    }

    const auto& target_status = sp_target_processor.Load(target_spm_path_s.c_str());
    if (!target_status.ok()) {
        std::string error_message = "Failed to load target SentencePiece model (" + target_spm_path_s + "): " + target_status.ToString();
        std::wcerr << L"[ERROR] " << std::wstring(error_message.begin(), error_message.end()) << std::endl;
        MessageBoxA(nullptr, error_message.c_str(), "Model Error", MB_OK | MB_ICONERROR);
        return false;
    }

    try {
        encoder_session = std::make_unique<Ort::Session>(env, encoder_model_path_w.c_str(), session_options);
        decoder_session = std::make_unique<Ort::Session>(env, decoder_model_path_w.c_str(), session_options);
    }
    catch (const Ort::Exception& e) {
        const char* char_what = e.what();
        int what_len = MultiByteToWideChar(CP_UTF8, 0, char_what, -1, nullptr, 0);
        std::wstring what_wstr(what_len, 0);
        MultiByteToWideChar(CP_UTF8, 0, char_what, -1, &what_wstr[0], what_len);
        std::wcerr << L"[ERROR] Failed to load ONNX models: " << what_wstr.c_str() << std::endl;
        MessageBoxW(nullptr, (L"Failed to load ONNX models from " + models_dir + L":\n" + what_wstr).c_str(), L"ONNX Error", MB_OK | MB_ICONERROR);
        return false;
    }
    return true;
}

std::wstring TranslateText(const std::wstring& input_text) {
    if (input_text.empty()) {
        return L"";
    }

    int utf8_len_needed = WideCharToMultiByte(CP_UTF8, 0, input_text.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (utf8_len_needed == 0) {
        std::wcerr << L"[ERROR] TranslateText: WideCharToMultiByte failed (1)." << std::endl;
        return L"[Translation Error: Input Conversion Failed]";
    }
    std::string utf8_input_str(utf8_len_needed - 1, '\0'); // Exclude null terminator for std::string
    WideCharToMultiByte(CP_UTF8, 0, input_text.c_str(), -1, &utf8_input_str[0], utf8_len_needed, nullptr, nullptr);

    std::vector<int32_t> input_ids_vec;
    sp_source_processor.Encode(utf8_input_str, &input_ids_vec);
    // The original code added <s> and </s>, let's assume model requires it.
    // BOS ID for many models is 0, EOS ID is often 2. Verify with your model.
    // input_ids_vec.insert(input_ids_vec.begin(), 0); // BOS
    // input_ids_vec.push_back(2); // EOS. Check model specifics. Often SentencePiece handles this.

    if (input_ids_vec.empty()) { // If after tokenization it's empty (e.g. only whitespace)
        return L"";
    }

    int64_t input_length = static_cast<int64_t>(input_ids_vec.size());
    std::vector<int64_t> input_shape = { 1, input_length };

    Ort::Value encoder_input_tensor = Ort::Value::CreateTensor<int32_t>(
        memory_info, input_ids_vec.data(), input_ids_vec.size(), input_shape.data(), input_shape.size());

    const char* encoder_input_names[] = { "input_ids" };
    const char* encoder_output_names[] = { "last_hidden_state" };

    std::vector<Ort::Value> encoder_outputs;
    try {
        encoder_outputs = encoder_session->Run(
            Ort::RunOptions{ nullptr },
            encoder_input_names, &encoder_input_tensor, 1,
            encoder_output_names, 1);
    }
    catch (const Ort::Exception& e) {
        std::wcerr << L"[ERROR] Encoder Run failed: " << e.what() << std::endl;
        return L"[Translation Error: Encoder Failed]";
    }


    Ort::Value& encoder_hidden_state = encoder_outputs[0];

    // Decoder loop
    std::vector<int32_t> decoder_input_ids = { 0 }; // Start with BOS token (<s> for many models)
    std::vector<int32_t> output_tokens;
    const int MAX_DECODE_STEPS = 128; // Max output length

    for (int step = 0; step < MAX_DECODE_STEPS; ++step) {
        std::vector<int64_t> decoder_input_shape = { 1, static_cast<int64_t>(decoder_input_ids.size()) };
        Ort::Value decoder_input_tensor = Ort::Value::CreateTensor<int32_t>(
            memory_info, decoder_input_ids.data(), decoder_input_ids.size(),
            decoder_input_shape.data(), decoder_input_shape.size());

        // Prepare inputs for the decoder session's Run method.
        // Input 1: decoder_input_tensor (current sequence of generated tokens for this step)
        // Input 2: encoder_hidden_state (output from the encoder, constant across decoder steps)

        // Get data from encoder_hidden_state (which is encoder_outputs[0]) to wrap it.
        // This wrapper Ort::Value will not own the data.
        auto& enc_out_val = encoder_outputs[0]; // This is the Ort::Value from the encoder
        auto enc_tensor_info = enc_out_val.GetTensorTypeAndShapeInfo();
        auto enc_tensor_element_type = enc_tensor_info.GetElementType();
        auto enc_tensor_shape = enc_tensor_info.GetShape();
        void* enc_tensor_data = enc_out_val.GetTensorMutableData<void>(); // Get raw data pointer

        // Create a new Ort::Value that wraps the encoder's output data.
        // This wrapper does not own the data; encoder_outputs[0] does.
        // The memory_info should be the same as the one used for encoder_outputs[0].
        // Assuming encoder_outputs[0] was created with memory_info (CPU).
        Ort::Value encoder_hidden_state_wrapper = Ort::Value::CreateTensor(
            memory_info, // Assuming data is on CPU, consistent with how encoder_outputs[0] would be.
            enc_tensor_data,
            enc_out_val.GetTensorTypeAndShapeInfo().GetElementCount() * Ort::GetTensorElementSize(enc_tensor_info), // Total data length in bytes
            enc_tensor_shape.data(),
            enc_tensor_shape.size(),
            enc_tensor_element_type
        );

        std::vector<Ort::Value> ort_decoder_inputs;
        ort_decoder_inputs.reserve(2);
        ort_decoder_inputs.emplace_back(std::move(decoder_input_tensor)); // Move the per-step decoder input
        ort_decoder_inputs.emplace_back(std::move(encoder_hidden_state_wrapper)); // Move the wrapper

        const char* decoder_input_names[] = { "input_ids", "encoder_hidden_states" };
        const char* decoder_output_names[] = { "logits" };

        std::vector<Ort::Value> decoder_outputs;
        try {
            decoder_outputs = decoder_session->Run(
                Ort::RunOptions{ nullptr },
                decoder_input_names, ort_decoder_inputs.data(), ort_decoder_inputs.size(), // Pass data() and size()
                decoder_output_names, 1);
        }
        catch (const Ort::Exception& e) {
            std::wcerr << L"[ERROR] Decoder Run failed: " << e.what() << std::endl;
            return L"[Translation Error: Decoder Failed]";
        }


        Ort::Value& logits_tensor = decoder_outputs[0];
        auto tensor_shape_info = logits_tensor.GetTensorTypeAndShapeInfo();
        auto shape = tensor_shape_info.GetShape(); // Should be [batch_size, sequence_length, vocab_size] e.g. [1, current_len, vocab]

        float* logits_data = logits_tensor.GetTensorMutableData<float>();
        int64_t vocab_size = shape[2];

        // Get logits for the last token
        float* last_token_logits = logits_data + (shape[1] - 1) * vocab_size;

        int32_t next_token_id = static_cast<int32_t>(std::distance(last_token_logits,
            std::max_element(last_token_logits, last_token_logits + vocab_size)));

        if (next_token_id == 2) { // EOS token ID (</s> for many models, verify for yours)
            break;
        }
        output_tokens.push_back(next_token_id);
        decoder_input_ids.push_back(next_token_id);

        if (output_tokens.size() >= MAX_DECODE_STEPS) break;
    }

    std::string decoded_text;
    sp_target_processor.Decode(output_tokens, &decoded_text);

    int wlen_needed = MultiByteToWideChar(CP_UTF8, 0, decoded_text.c_str(), -1, nullptr, 0);
    if (wlen_needed == 0 && !decoded_text.empty()) { // Check if decoded_text is empty to avoid error on empty string
        std::wcerr << L"[ERROR] TranslateText: MultiByteToWideChar failed (2)." << std::endl;
        return L"[Translation Error: Output Conversion Failed]";
    }
    if (wlen_needed == 0 && decoded_text.empty()) {
        return L""; // Successfully decoded to an empty string
    }

    std::wstring final_result(wlen_needed - 1, L'\0'); // Exclude null terminator
    MultiByteToWideChar(CP_UTF8, 0, decoded_text.c_str(), -1, &final_result[0], wlen_needed);

    return final_result;
}

INT_PTR CALLBACK MainDlgProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_INITDIALOG:
        CheckRadioButton(hDlg, IDC_RADIO_FULLSCREEN, IDC_RADIO_REGION, IDC_RADIO_FULLSCREEN);
        return TRUE;
    case WM_COMMAND:
        switch (LOWORD(wParam))
        {
        case IDOK:
            g_fullscreen_mode = (IsDlgButtonChecked(hDlg, IDC_RADIO_FULLSCREEN) == BST_CHECKED);
            EndDialog(hDlg, IDOK);
            return TRUE;
        case IDCANCEL:
            EndDialog(hDlg, IDCANCEL);
            return TRUE;
        }
        break;
    }
    return FALSE;
}

RECT SelectScreenRegion()
{
    HWND hSelectionOverlay = CreateWindowExW(
        WS_EX_TOPMOST | WS_EX_LAYERED | WS_EX_TRANSPARENT,
        L"STATIC", L"Drag to select region", WS_POPUP | WS_BORDER, // Added WS_BORDER for visibility
        0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN),
        nullptr, nullptr, g_hinstance, nullptr);

    // Make it slightly transparent so user can see through
    SetLayeredWindowAttributes(hSelectionOverlay, 0, 180, LWA_ALPHA);
    ShowWindow(hSelectionOverlay, SW_SHOW);
    UpdateWindow(hSelectionOverlay); // Ensure it's painted

    POINT start_pt{}, end_pt{};
    bool selecting_region = false;
    HDC hdc = GetDC(hSelectionOverlay); // For drawing selection rectangle
    HPEN hPen = CreatePen(PS_SOLID, 2, RGB(255, 0, 0)); // Red pen for selection rectangle
    SelectObject(hdc, GetStockObject(NULL_BRUSH)); // Transparent fill
    SelectObject(hdc, hPen);

    // SetCapture(hSelectionOverlay); // Capture mouse input

    MSG msg;
    while (true)
    {
        // Standard message loop for this modal selection
        if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        if (GetAsyncKeyState(VK_LBUTTON) & 0x8000) // LBUTTON down
        {
            if (!selecting_region)
            {
                GetCursorPos(&start_pt);
                ScreenToClient(hSelectionOverlay, &start_pt); // Convert to window coordinates for drawing
                end_pt = start_pt; // Initialize end_pt
                selecting_region = true;
                SetCapture(hSelectionOverlay); // Capture mouse to this window
            }
            else {
                GetCursorPos(&end_pt);
                ScreenToClient(hSelectionOverlay, &end_pt);
                // Redraw the selection rectangle
                InvalidateRect(hSelectionOverlay, NULL, TRUE); // Invalidate to trigger WM_PAINT
                UpdateWindow(hSelectionOverlay); // Force paint now

                // Or draw directly if WM_PAINT is too slow for live feedback
                // RECT currentRect = {min(start_pt.x, end_pt.x), min(start_pt.y, end_pt.y), max(start_pt.x, end_pt.x), max(start_pt.y, end_pt.y)};
                // InvalidateRect(hSelectionOverlay, NULL, TRUE); // Clear previous
                // Rectangle(hdc, currentRect.left, currentRect.top, currentRect.right, currentRect.bottom);
            }
        }
        else // LBUTTON up or not pressed
        {
            if (selecting_region) // Means LBUTTON was just released
            {
                ReleaseCapture();
                GetCursorPos(&end_pt); // Final position
                break;
            }
        }
        if (GetAsyncKeyState(VK_ESCAPE) & 0x8000) // Allow canceling selection
        {
            ReleaseCapture();
            DestroyWindow(hSelectionOverlay);
            DeleteObject(hPen);
            ReleaseDC(hSelectionOverlay, hdc);
            return RECT{ 0,0,0,0 }; // Indicate cancellation or invalid region
        }
        Sleep(10); // Don't peg CPU
    }

    ReleaseCapture(); // Just in case
    DeleteObject(hPen);
    ReleaseDC(hSelectionOverlay, hdc);
    DestroyWindow(hSelectionOverlay);

    // Convert back to screen coordinates if they were client
    // GetCursorPos gives screen coords, so start_pt & end_pt from there are screen.
    // The issue was if I used ClientToScreen before GetCursorPos.
    // The original GetCursorPos(&start) and GetCursorPos(&end) are already screen coordinates.

    // The start and end points are already in screen coordinates.
    return RECT{
        min(start_pt.x, end_pt.x),
        min(start_pt.y, end_pt.y),
        max(start_pt.x, end_pt.x),
        max(start_pt.y, end_pt.y)
    };
}


// Interface for IBufferByteAccess
#include <robuffer.h> // For Windows::Storage::Streams::IBufferByteAccess

SoftwareBitmap CaptureScreen(int x, int y, int width, int height)
{
    if (width <= 0 || height <= 0) return nullptr;

    HDC dcScreen = GetDC(nullptr);
    HDC dcMem = CreateCompatibleDC(dcScreen);
    HBITMAP bmp = CreateCompatibleBitmap(dcScreen, width, height);
    HGDIOBJ oldBmp = SelectObject(dcMem, bmp); // Store old object

    BitBlt(dcMem, 0, 0, width, height, dcScreen, x, y, SRCCOPY);

    BITMAPINFOHEADER bi = { sizeof(bi), width, -height, 1, 32, BI_RGB }; // -height for top-down DIB
    std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 4);
    GetDIBits(dcMem, bmp, 0, height, pixels.data(), reinterpret_cast<BITMAPINFO*>(&bi), DIB_RGB_COLORS);

    SelectObject(dcMem, oldBmp); // Restore old bitmap
    ReleaseDC(nullptr, dcScreen);
    DeleteDC(dcMem);
    DeleteObject(bmp);

    try {
        SoftwareBitmap softwareBitmap(BitmapPixelFormat::Bgra8, width, height, BitmapAlphaMode::Ignore);

        BitmapBuffer buffer = softwareBitmap.LockBuffer(BitmapBufferAccessMode::Write);
        IMemoryBufferReference reference = buffer.CreateReference();

        uint8_t* destPixels = nullptr;
        uint32_t capacity = 0;

        // Query for IBufferByteAccess to get raw pointer to buffer
        auto byteAccess = reference.as<::Windows::Storage::Streams::IBufferByteAccess>();
        winrt::check_hresult(byteAccess->Buffer(&destPixels)); // Get pointer to buffer
        capacity = reference.Capacity(); // Get buffer capacity

        if (capacity >= pixels.size()) {
            memcpy(destPixels, pixels.data(), pixels.size());
        }
        else {
            std::wcerr << L"CaptureScreen: SoftwareBitmap buffer capacity (" << capacity
                << L") is less than pixel data size (" << pixels.size() << L")." << std::endl;
            reference.Close();
            buffer.Close();
            return nullptr;
        }

        reference.Close();
        buffer.Close();

        return softwareBitmap;

    }
    catch (winrt::hresult_error const& ex) {
        std::wcerr << L"CaptureScreen: Failed to create or populate SoftwareBitmap: " << ex.message().c_str() << std::endl;
        return nullptr;
    }
}

LRESULT CALLBACK OverlayWndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
    switch (msg)
    {
    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);
        SetBkMode(hdc, TRANSPARENT);
        SetTextColor(hdc, RGB(255, 255, 0)); // Yellow text
        RECT client_rect;
        GetClientRect(hwnd, &client_rect);

        HFONT hFont = CreateFontW(24, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
            OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY,
            DEFAULT_PITCH | FF_SWISS, L"Arial");
        HFONT oldFont = (HFONT)SelectObject(hdc, hFont);

        DrawTextW(hdc, g_current_overlay_text.c_str(), -1, &client_rect, DT_CENTER | DT_VCENTER | DT_WORDBREAK | DT_NOCLIP);

        SelectObject(hdc, oldFont);
        DeleteObject(hFont);
        EndPaint(hwnd, &ps);
        return 0;
    }
    // WM_DESTROY is handled by DefWindowProc which is fine.
    // No PostQuitMessage here to avoid killing main app.
    }
    return DefWindowProc(hwnd, msg, wp, lp);
}

void RegisterOverlayWindowClass(HINSTANCE hInstance) {
    WNDCLASSW wc = { 0 };
    wc.lpfnWndProc = OverlayWndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = OVERLAY_WINDOW_CLASS;
    wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH); // Will be made transparent
    wc.style = CS_HREDRAW | CS_VREDRAW;
    if (!RegisterClassW(&wc)) {
        MessageBoxW(nullptr, L"Failed to register overlay window class.", L"Error", MB_OK | MB_ICONERROR);
    }
}

void UnregisterOverlayWindowClass(HINSTANCE hInstance) {
    UnregisterClassW(OVERLAY_WINDOW_CLASS, hInstance);
}

void CreateOrUpdateOverlayWindow(const std::wstring& text, const RECT& target_ocr_region)
{
    g_current_overlay_text = text;

    if (!g_overlay_hwnd) {
        // Define overlay size and position (e.g., below the OCR'd region or fixed)
        int overlay_width = 600;
        int overlay_height = 100;
        int overlay_x = target_ocr_region.left;
        int overlay_y = target_ocr_region.bottom + 5; // 5px below the region

        // Ensure overlay is within screen bounds
        HMONITOR hMonitor = MonitorFromRect(&target_ocr_region, MONITOR_DEFAULTTONEAREST);
        MONITORINFO monitorInfo = { sizeof(monitorInfo) };
        if (GetMonitorInfo(hMonitor, &monitorInfo)) {
            if (overlay_x + overlay_width > monitorInfo.rcWork.right) {
                overlay_x = monitorInfo.rcWork.right - overlay_width;
            }
            if (overlay_y + overlay_height > monitorInfo.rcWork.bottom) {
                overlay_y = monitorInfo.rcWork.bottom - overlay_height;
                if (overlay_y < monitorInfo.rcWork.top) overlay_y = target_ocr_region.top - overlay_height - 5; // try above
            }
            if (overlay_x < monitorInfo.rcWork.left) overlay_x = monitorInfo.rcWork.left;
            if (overlay_y < monitorInfo.rcWork.top) overlay_y = monitorInfo.rcWork.top;

        }


        g_overlay_hwnd = CreateWindowExW(
            WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_NOACTIVATE, // WS_EX_NOACTIVATE so it doesn't steal focus
            OVERLAY_WINDOW_CLASS,
            L"Translation Overlay", // Window title (not visible usually for such overlays)
            WS_POPUP, // No border, no caption
            overlay_x, overlay_y, overlay_width, overlay_height,
            nullptr, nullptr, g_hinstance, nullptr);

        if (!g_overlay_hwnd) {
            std::wcerr << L"Failed to create overlay window. Error: " << GetLastError() << std::endl;
            return;
        }
        // Set transparency: 0 for fully transparent, 255 for opaque.
        // Key color transparency: SetLayeredWindowAttributes(g_overlay_hwnd, RGB(0,0,0), 0, LWA_COLORKEY);
        // Alpha transparency:
        SetLayeredWindowAttributes(g_overlay_hwnd, 0, 220, LWA_ALPHA); // 220 for semi-transparent
        ShowWindow(g_overlay_hwnd, SW_SHOWNA); // SW_SHOWNA to show without activating
        UpdateWindow(g_overlay_hwnd);
    }
    else {
        // Update position if needed (e.g. if target_ocr_region moves)
        // For now, we just update text.
        InvalidateRect(g_overlay_hwnd, nullptr, TRUE); // Trigger repaint
        UpdateWindow(g_overlay_hwnd);
    }
}


int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
    _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPWSTR    lpCmdLine,
    _In_ int       nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    g_hinstance = hInstance;
    winrt::init_apartment(apartment_type::single_threaded);

    RegisterOverlayWindowClass(hInstance);

    if (!InitTranslationEngine())
    {
        std::wcerr << L"[FATAL] Translation engine initialization failed." << std::endl;
        MessageBoxW(nullptr, L"Translation engine failed to initialize. Check model paths and dependencies.", L"Initialization Error", MB_OK | MB_ICONERROR);
        UnregisterOverlayWindowClass(hInstance);
        winrt::uninit_apartment();
        return -1;
    }

    INT_PTR dlg_result = DialogBoxParamW(
        hInstance,
        MAKEINTRESOURCE(IDD_MAIN_DIALOG),
        nullptr,
        MainDlgProc,
        0); // lParam for DialogBoxParamW

    if (dlg_result != IDOK) {
        UnregisterOverlayWindowClass(hInstance);
        winrt::uninit_apartment();
        return 0;
    }

    RECT capture_region{};
    if (g_fullscreen_mode)
    {
        capture_region = { 0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN) };
    }
    else
    {
        capture_region = SelectScreenRegion();
        if ((capture_region.right - capture_region.left) < 10 || (capture_region.bottom - capture_region.top) < 10)
        {
            MessageBoxW(nullptr, L"Selected region is too small.", L"Region Error", MB_OK | MB_ICONWARNING);
            UnregisterOverlayWindowClass(hInstance);
            winrt::uninit_apartment();
            return 0;
        }
    }

    OcrEngine engine = nullptr;
    try {
        engine = OcrEngine::TryCreateFromLanguage(Language(L"en"));
    }
    catch (winrt::hresult_error const& ex) {
        std::wcerr << L"[FATAL] Failed to create OCR Engine: " << ex.message().c_str() << std::endl;
    }

    if (!engine)
    {
        std::wcerr << L"[FATAL] OCR engine could not be initialized. Is the English language pack installed?" << std::endl;
        MessageBoxW(nullptr, L"OCR engine (English) failed to initialize. Please ensure the English language pack for OCR is installed in Windows settings.", L"OCR Error", MB_OK | MB_ICONERROR);
        UnregisterOverlayWindowClass(hInstance);
        winrt::uninit_apartment();
        return -1;
    }

    std::wstring last_ocr_text;
    bool keep_running = true;

    // Pre-create overlay if not already, or ensure it's ready
    // CreateOrUpdateOverlayWindow(L"Initializing...", capture_region); // Initial placeholder

    MSG msg = {};
    while (keep_running)
    {
        // Process messages for the main thread (e.g., for dialogs, future UI elements)
        // And also for the overlay window if its messages are dispatched to this thread.
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            if (msg.message == WM_QUIT) {
                keep_running = false;
                break;
            }
            // If we had other windows or modeless dialogs, we'd dispatch here.
            // For now, the overlay is simple.
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        if (!keep_running) break;


        SoftwareBitmap software_bitmap = CaptureScreen(
            capture_region.left,
            capture_region.top,
            capture_region.right - capture_region.left,
            capture_region.bottom - capture_region.top);

        if (software_bitmap) {
            OcrResult ocr_result = nullptr;
            try {
                ocr_result = engine.RecognizeAsync(software_bitmap).get();
            }
            catch (winrt::hresult_error const& ex) {
                std::wcerr << L"OCR RecognizeAsync error: " << ex.message().c_str() << std::endl;
                // Maybe show a temporary error on overlay or log
            }

            if (ocr_result && !ocr_result.Text().empty()) {
                std::wstring current_text = ocr_result.Text().c_str();
                if (current_text != last_ocr_text) {
                    last_ocr_text = current_text;
                    std::wstring translated_text = TranslateText(current_text);
                    if (!translated_text.empty()) {
                        CreateOrUpdateOverlayWindow(translated_text, capture_region);
                    }
                    else if (translated_text.empty() && !current_text.empty()) {
                        // If translation is empty but source wasn't, maybe show source or "..."
                        CreateOrUpdateOverlayWindow(L"...", capture_region);
                    }
                }
            }
        }
        else {
            // Handle capture failure, maybe log or show error on overlay
            // CreateOrUpdateOverlayWindow(L"[Capture Error]", capture_region);
        }

        if (GetAsyncKeyState(VK_ESCAPE) & 0x8000) {
            keep_running = false;
        }
        Sleep(500); // Main loop polling interval
    }

    if (g_overlay_hwnd) {
        DestroyWindow(g_overlay_hwnd);
        g_overlay_hwnd = nullptr;
    }

    UnregisterOverlayWindowClass(hInstance);
    winrt::uninit_apartment();
    return 0;
}
