#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include <sentencepiece_processor.h>
#include <dwmapi.h>
#include <iostream>
#include <thread>
#include <vector>
#include <string>
#include <memory>
#include <filesystem>
#include <optional>
#include <algorithm>
#include <stdexcept>
#include <format>
#include <windows.h>
#include <shellapi.h> // For SHGetKnownFolderPath
#include <ShlObj_core.h> // For FOLDERID_RoamingAppData

#include "resource.h"

#include <winrt/base.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Globalization.h>
#include <winrt/Windows.Media.Ocr.h>
#include <winrt/Windows.Graphics.Imaging.h>
#include <winrt/Windows.Storage.Streams.h>

#pragma comment(lib, "Dwmapi.lib")
#pragma comment(lib, "WindowsApp.lib")
#pragma comment(lib, "Shell32.lib") // For SHGetKnownFolderPath

using namespace winrt;
using namespace winrt::Windows::Foundation;
using namespace winrt::Windows::Globalization;
using namespace winrt::Windows::Media::Ocr;
using namespace winrt::Windows::Graphics::Imaging;
using namespace winrt::Windows::Storage::Streams;

// --- Modern C++ Helper Functions ---

// UTF-16 (wstring) -> UTF-8 (string)
inline std::string wstring_to_utf8(const std::wstring& wstr) {
    if (wstr.empty()) return {};
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr.data(), static_cast<int>(wstr.size()), nullptr, 0, nullptr, nullptr);
    std::string strTo(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.data(), static_cast<int>(wstr.size()), &strTo[0], size_needed, nullptr, nullptr);
    return strTo;
}

// UTF-8 (string) -> UTF-16 (wstring)
inline std::wstring utf8_to_wstring(const std::string& str) {
    if (str.empty()) return {};
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), nullptr, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), &wstrTo[0], size_needed);
    return wstrTo;
}

// Modern error message box
inline void show_message_box(const std::wstring& text, const std::wstring& caption = L"Hata", UINT type = MB_OK | MB_ICONERROR) {
    MessageBoxW(nullptr, text.c_str(), caption.c_str(), type);
}

// --- ONNX Runtime Helper ---

namespace Ort {
    inline size_t GetTensorElementSize(ONNXTensorElementDataType type) {
        switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return sizeof(float);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return sizeof(uint8_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return sizeof(int8_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  return sizeof(uint16_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:   return sizeof(int16_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return sizeof(int32_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return sizeof(int64_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:  return sizeof(std::string);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    return sizeof(bool);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return sizeof(uint16_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  return sizeof(double);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  return sizeof(uint32_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:  return sizeof(uint64_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return sizeof(uint16_t);
        default:
            throw std::runtime_error("Unsupported ONNX tensor element data type in GetTensorElementSize: " + std::to_string(type));
        }
    }
}

// --- Global State ---

Ort::SessionOptions session_options;
Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
std::unique_ptr<Ort::Session> encoder_session;
std::unique_ptr<Ort::Session> decoder_session;
sentencepiece::SentencePieceProcessor sp_source_processor;
sentencepiece::SentencePieceProcessor sp_target_processor;
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ocr-translator-env");

bool g_fullscreen_mode = true;
std::wstring g_current_overlay_text;
HWND g_overlay_hwnd = nullptr;
const wchar_t OVERLAY_WINDOW_CLASS[] = L"OcrTranslationOverlayWindowClass";
HINSTANCE g_hinstance = nullptr;

// --- Forward Declarations ---
LRESULT CALLBACK OverlayWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
void RegisterOverlayWindowClass(HINSTANCE hInstance);
void UnregisterOverlayWindowClass(HINSTANCE hInstance);
std::filesystem::path GetModelsDirectoryPath();
void CreateOrUpdateOverlayWindow(const std::wstring& text, const RECT& target_region);

// --- Implementation ---

std::filesystem::path GetModelsDirectoryPath() {
    std::vector<wchar_t> exePathBuffer(MAX_PATH);
    DWORD len = GetModuleFileNameW(nullptr, exePathBuffer.data(), static_cast<DWORD>(exePathBuffer.size()));
    std::wstring exePath(exePathBuffer.data(), len);
    auto path = std::filesystem::path(exePath).parent_path() / L"models";
    return path;
}

bool InitTranslationEngine() {
    session_options.SetIntraOpNumThreads(static_cast<int>(std::thread::hardware_concurrency()));
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    auto models_dir = GetModelsDirectoryPath();
    auto source_spm_path = models_dir / L"source.spm";
    auto target_spm_path = models_dir / L"target.spm";
    auto encoder_model_path = models_dir / L"encoder_model.onnx";
    auto decoder_model_path = models_dir / L"decoder_model.onnx";

    // SentencePiece: wstring -> utf8 string
    auto source_spm_path_s = wstring_to_utf8(source_spm_path.wstring());
    auto target_spm_path_s = wstring_to_utf8(target_spm_path.wstring());

    auto source_status = sp_source_processor.Load(source_spm_path_s.c_str());
    if (!source_status.ok()) {
        std::wstring error_message = utf8_to_wstring("Failed to load source SentencePiece model (" + source_spm_path_s + "): " + source_status.ToString());
        show_message_box(error_message, L"Model Error");
        return false;
    }
    auto target_status = sp_target_processor.Load(target_spm_path_s.c_str());
    if (!target_status.ok()) {
        std::wstring error_message = utf8_to_wstring("Failed to load target SentencePiece model (" + target_spm_path_s + "): " + target_status.ToString());
        show_message_box(error_message, L"Model Error");
        return false;
    }

    try {
        encoder_session = std::make_unique<Ort::Session>(env, encoder_model_path.c_str(), session_options);
        decoder_session = std::make_unique<Ort::Session>(env, decoder_model_path.c_str(), session_options);
    } catch (const Ort::Exception& e) {
        show_message_box(utf8_to_wstring(std::format("Failed to load ONNX models from {}: {}", models_dir.wstring(), e.what())), L"ONNX Error");
        return false;
    }
    return true;
}

std::wstring TranslateText(const std::wstring& input_text) {
    if (input_text.empty()) return L"";

    auto utf8_input_str = wstring_to_utf8(input_text);
    std::vector<int32_t> input_ids_vec;
    sp_source_processor.Encode(utf8_input_str, &input_ids_vec);

    if (input_ids_vec.empty()) return L"";

    int64_t input_length = static_cast<int64_t>(input_ids_vec.size());
    std::vector<int64_t> input_shape = { 1, input_length };

    auto encoder_input_tensor = Ort::Value::CreateTensor<int32_t>(
        memory_info, input_ids_vec.data(), input_ids_vec.size(),
        input_shape.data(), input_shape.size());

    const char* encoder_input_names[] = { "input_ids" };
    const char* encoder_output_names[] = { "last_hidden_state" };

    std::vector<Ort::Value> encoder_outputs;
    try {
        encoder_outputs = encoder_session->Run(
            Ort::RunOptions{ nullptr },
            encoder_input_names, &encoder_input_tensor, 1,
            encoder_output_names, 1);
    } catch (const Ort::Exception& e) {
        return L"[Translation Error: Encoder Failed]";
    }

    auto& encoder_hidden_state = encoder_outputs[0];

    std::vector<int32_t> decoder_input_ids = { 0 }; // BOS
    std::vector<int32_t> output_tokens;
    constexpr int MAX_DECODE_STEPS = 128;

    for (int step = 0; step < MAX_DECODE_STEPS; ++step) {
        std::vector<int64_t> decoder_input_shape = { 1, static_cast<int64_t>(decoder_input_ids.size()) };
        auto decoder_input_tensor = Ort::Value::CreateTensor<int32_t>(
            memory_info, decoder_input_ids.data(), decoder_input_ids.size(),
            decoder_input_shape.data(), decoder_input_shape.size());

        auto& enc_out_val = encoder_outputs[0];
        auto enc_tensor_info = enc_out_val.GetTensorTypeAndShapeInfo();
        void* enc_tensor_data = enc_out_val.GetTensorMutableData<void>();
        auto enc_tensor_shape = enc_tensor_info.GetShape();
        auto enc_tensor_element_type = enc_tensor_info.GetElementType();

        auto encoder_hidden_state_wrapper = Ort::Value::CreateTensor(
            memory_info,
            enc_tensor_data,
            enc_out_val.GetTensorTypeAndShapeInfo().GetElementCount() * Ort::GetTensorElementSize(enc_tensor_info.GetElementType()),
            enc_tensor_shape.data(),
            enc_tensor_shape.size(),
            enc_tensor_element_type
        );

        std::vector<Ort::Value> ort_decoder_inputs;
        ort_decoder_inputs.reserve(2);
        ort_decoder_inputs.emplace_back(std::move(decoder_input_tensor));
        ort_decoder_inputs.emplace_back(std::move(encoder_hidden_state_wrapper));

        const char* decoder_input_names[] = { "input_ids", "encoder_hidden_states" };
        const char* decoder_output_names[] = { "logits" };

        std::vector<Ort::Value> decoder_outputs;
        try {
            decoder_outputs = decoder_session->Run(
                Ort::RunOptions{ nullptr },
                decoder_input_names, ort_decoder_inputs.data(), ort_decoder_inputs.size(),
                decoder_output_names, 1);
        } catch (const Ort::Exception&) {
            return L"[Translation Error: Decoder Failed]";
        }

        auto& logits_tensor = decoder_outputs[0];
        auto tensor_shape_info = logits_tensor.GetTensorTypeAndShapeInfo();
        auto shape = tensor_shape_info.GetShape();
        float* logits_data = logits_tensor.GetTensorMutableData<float>();
        int64_t vocab_size = shape[2];

        float* last_token_logits = logits_data + (shape[1] - 1) * vocab_size;
        int32_t next_token_id = static_cast<int32_t>(std::distance(last_token_logits,
            std::max_element(last_token_logits, last_token_logits + vocab_size)));

        if (next_token_id == 2) break; // EOS
        output_tokens.push_back(next_token_id);
        decoder_input_ids.push_back(next_token_id);

        if (output_tokens.size() >= MAX_DECODE_STEPS) break;
    }

    std::string decoded_text;
    sp_target_processor.Decode(output_tokens, &decoded_text);

    return utf8_to_wstring(decoded_text);
}

INT_PTR CALLBACK MainDlgProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM) {
    switch (message) {
    case WM_INITDIALOG:
        CheckRadioButton(hDlg, IDC_RADIO_FULLSCREEN, IDC_RADIO_REGION, IDC_RADIO_FULLSCREEN);
        return TRUE;
    case WM_COMMAND:
        switch (LOWORD(wParam)) {
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

RECT SelectScreenRegion() {
    HWND hSelectionOverlay = CreateWindowExW(
        WS_EX_TOPMOST | WS_EX_LAYERED | WS_EX_TRANSPARENT,
        L"STATIC", L"Drag to select region", WS_POPUP | WS_BORDER,
        0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN),
        nullptr, nullptr, g_hinstance, nullptr);

    SetLayeredWindowAttributes(hSelectionOverlay, 0, 180, LWA_ALPHA);
    ShowWindow(hSelectionOverlay, SW_SHOW);
    UpdateWindow(hSelectionOverlay);

    POINT start_pt{}, end_pt{};
    bool selecting_region = false;
    HDC hdc = GetDC(hSelectionOverlay);
    auto hPen = std::unique_ptr<HPEN, decltype(&DeleteObject)>(
        CreatePen(PS_SOLID, 2, RGB(255, 0, 0)), DeleteObject);
    SelectObject(hdc, GetStockObject(NULL_BRUSH));
    SelectObject(hdc, hPen.get());

    MSG msg;
    while (true) {
        if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        if (GetAsyncKeyState(VK_LBUTTON) & 0x8000) {
            if (!selecting_region) {
                GetCursorPos(&start_pt);
                ScreenToClient(hSelectionOverlay, &start_pt);
                end_pt = start_pt;
                selecting_region = true;
                SetCapture(hSelectionOverlay);
            } else {
                GetCursorPos(&end_pt);
                ScreenToClient(hSelectionOverlay, &end_pt);
                InvalidateRect(hSelectionOverlay, NULL, TRUE);
                UpdateWindow(hSelectionOverlay);
            }
        } else {
            if (selecting_region) {
                ReleaseCapture();
                GetCursorPos(&end_pt);
                break;
            }
        }
        if (GetAsyncKeyState(VK_ESCAPE) & 0x8000) {
            ReleaseCapture();
            DestroyWindow(hSelectionOverlay);
            ReleaseDC(hSelectionOverlay, hdc);
            return RECT{ 0,0,0,0 };
        }
        Sleep(10);
    }

    ReleaseCapture();
    ReleaseDC(hSelectionOverlay, hdc);
    DestroyWindow(hSelectionOverlay);

    return RECT{
        min(start_pt.x, end_pt.x),
        min(start_pt.y, end_pt.y),
        max(start_pt.x, end_pt.x),
        max(start_pt.y, end_pt.y)
    };
}

// Interface for IBufferByteAccess
#include <robuffer.h> // For Windows::Storage::Streams::IBufferByteAccess

SoftwareBitmap CaptureScreen(int x, int y, int width, int height) {
    if (width <= 0 || height <= 0) return nullptr;

    HDC dcScreen = GetDC(nullptr);
    HDC dcMem = CreateCompatibleDC(dcScreen);
    HBITMAP bmp = CreateCompatibleBitmap(dcScreen, width, height);
    HGDIOBJ oldBmp = SelectObject(dcMem, bmp);

    BitBlt(dcMem, 0, 0, width, height, dcScreen, x, y, SRCCOPY);

    BITMAPINFOHEADER bi = { sizeof(bi), width, -height, 1, 32, BI_RGB };
    std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 4);
    GetDIBits(dcMem, bmp, 0, height, pixels.data(), reinterpret_cast<BITMAPINFO*>(&bi), DIB_RGB_COLORS);

    SelectObject(dcMem, oldBmp);
    ReleaseDC(nullptr, dcScreen);
    DeleteDC(dcMem);
    DeleteObject(bmp);

    try {
        SoftwareBitmap softwareBitmap(BitmapPixelFormat::Bgra8, width, height, BitmapAlphaMode::Ignore);

        BitmapBuffer buffer = softwareBitmap.LockBuffer(BitmapBufferAccessMode::Write);
        IMemoryBufferReference reference = buffer.CreateReference();

        uint8_t* destPixels = nullptr;
        uint32_t capacity = 0;

        auto byteAccess = reference.as<::Windows::Storage::Streams::IBufferByteAccess>();
        winrt::check_hresult(byteAccess->Buffer(&destPixels));
        capacity = reference.Capacity();

        if (capacity >= pixels.size()) {
            std::copy(pixels.begin(), pixels.end(), destPixels);
        } else {
            reference.Close();
            buffer.Close();
            return nullptr;
        }

        reference.Close();
        buffer.Close();
        return softwareBitmap;

    } catch (winrt::hresult_error const&) {
        return nullptr;
    }
}

LRESULT CALLBACK OverlayWndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);
        SetBkMode(hdc, TRANSPARENT);
        SetTextColor(hdc, RGB(255, 255, 0));
        RECT client_rect;
        GetClientRect(hwnd, &client_rect);

        HFONT hFont = CreateFontW(24, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
            OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY,
            DEFAULT_PITCH | FF_SWISS, L"Arial");
        HFONT oldFont = static_cast<HFONT>(SelectObject(hdc, hFont));
        DrawTextW(hdc, g_current_overlay_text.c_str(), -1, &client_rect, DT_CENTER | DT_VCENTER | DT_WORDBREAK | DT_NOCLIP);
        SelectObject(hdc, oldFont);
        DeleteObject(hFont);
        EndPaint(hwnd, &ps);
        return 0;
    }
    }
    return DefWindowProc(hwnd, msg, wp, lp);
}

void RegisterOverlayWindowClass(HINSTANCE hInstance) {
    WNDCLASSW wc = { 0 };
    wc.lpfnWndProc = OverlayWndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = OVERLAY_WINDOW_CLASS;
    wc.hbrBackground = static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
    wc.style = CS_HREDRAW | CS_VREDRAW;
    if (!RegisterClassW(&wc)) {
        show_message_box(L"Failed to register overlay window class.", L"Error");
    }
}

void UnregisterOverlayWindowClass(HINSTANCE hInstance) {
    UnregisterClassW(OVERLAY_WINDOW_CLASS, hInstance);
}

void CreateOrUpdateOverlayWindow(const std::wstring& text, const RECT& target_ocr_region) {
    g_current_overlay_text = text;

    if (!g_overlay_hwnd) {
        int overlay_width = 600;
        int overlay_height = 100;
        int overlay_x = target_ocr_region.left;
        int overlay_y = target_ocr_region.bottom + 5;

        HMONITOR hMonitor = MonitorFromRect(&target_ocr_region, MONITOR_DEFAULTTONEAREST);
        MONITORINFO monitorInfo = { sizeof(monitorInfo) };
        if (GetMonitorInfo(hMonitor, &monitorInfo)) {
            if (overlay_x + overlay_width > monitorInfo.rcWork.right)
                overlay_x = monitorInfo.rcWork.right - overlay_width;
            if (overlay_y + overlay_height > monitorInfo.rcWork.bottom)
                overlay_y = monitorInfo.rcWork.bottom - overlay_height;
            if (overlay_x < monitorInfo.rcWork.left) overlay_x = monitorInfo.rcWork.left;
            if (overlay_y < monitorInfo.rcWork.top) overlay_y = monitorInfo.rcWork.top;
        }

        g_overlay_hwnd = CreateWindowExW(
            WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_NOACTIVATE,
            OVERLAY_WINDOW_CLASS,
            L"Translation Overlay",
            WS_POPUP,
            overlay_x, overlay_y, overlay_width, overlay_height,
            nullptr, nullptr, g_hinstance, nullptr);

        if (!g_overlay_hwnd) {
            return;
        }
        SetLayeredWindowAttributes(g_overlay_hwnd, 0, 220, LWA_ALPHA);
        ShowWindow(g_overlay_hwnd, SW_SHOWNA);
        UpdateWindow(g_overlay_hwnd);
    } else {
        InvalidateRect(g_overlay_hwnd, nullptr, TRUE);
        UpdateWindow(g_overlay_hwnd);
    }
}

int APIENTRY wWinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE, _In_ LPWSTR, _In_ int) {
    g_hinstance = hInstance;
    winrt::init_apartment(apartment_type::single_threaded);

    RegisterOverlayWindowClass(hInstance);

    if (!InitTranslationEngine()) {
        show_message_box(L"Translation engine failed to initialize. Check model paths and dependencies.", L"Initialization Error");
        UnregisterOverlayWindowClass(hInstance);
        winrt::uninit_apartment();
        return -1;
    }

    INT_PTR dlg_result = DialogBoxParamW(
        hInstance,
        MAKEINTRESOURCE(IDD_MAIN_DIALOG),
        nullptr,
        MainDlgProc,
        0);

    if (dlg_result != IDOK) {
        UnregisterOverlayWindowClass(hInstance);
        winrt::uninit_apartment();
        return 0;
    }

    RECT capture_region{};
    if (g_fullscreen_mode) {
        capture_region = { 0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN) };
    } else {
        capture_region = SelectScreenRegion();
        if ((capture_region.right - capture_region.left) < 10 || (capture_region.bottom - capture_region.top) < 10) {
            show_message_box(L"Selected region is too small.", L"Region Error", MB_OK | MB_ICONWARNING);
            UnregisterOverlayWindowClass(hInstance);
            winrt::uninit_apartment();
            return 0;
        }
    }

    OcrEngine engine = nullptr;
    try {
        engine = OcrEngine::TryCreateFromLanguage(Language(L"en"));
    } catch (winrt::hresult_error const&) {}

    if (!engine) {
        show_message_box(L"OCR engine (English) failed to initialize. Please ensure the English language pack for OCR is installed in Windows settings.", L"OCR Error");
        UnregisterOverlayWindowClass(hInstance);
        winrt::uninit_apartment();
        return -1;
    }

    std::wstring last_ocr_text;
    bool keep_running = true;
    MSG msg = {};
    while (keep_running) {
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                keep_running = false;
                break;
            }
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
            } catch (winrt::hresult_error const&) {}

            if (ocr_result && !ocr_result.Text().empty()) {
                std::wstring current_text = ocr_result.Text().c_str();
                if (current_text != last_ocr_text) {
                    last_ocr_text = current_text;
                    std::wstring translated_text = TranslateText(current_text);
                    if (!translated_text.empty()) {
                        CreateOrUpdateOverlayWindow(translated_text, capture_region);
                    } else if (translated_text.empty() && !current_text.empty()) {
                        CreateOrUpdateOverlayWindow(L"...", capture_region);
                    }
                }
            }
        }

        if (GetAsyncKeyState(VK_ESCAPE) & 0x8000) {
            keep_running = false;
        }
        Sleep(500);
    }

    if (g_overlay_hwnd) {
        DestroyWindow(g_overlay_hwnd);
        g_overlay_hwnd = nullptr;
    }
    UnregisterOverlayWindowClass(hInstance);
    winrt::uninit_apartment();
    return 0;
}
