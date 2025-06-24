#include <windows.h>
#include <dwmapi.h>
#include <iostream>
#include <thread>
#include <vector>
#include <string>
#include <conio.h>

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

using namespace winrt;
using namespace Windows::Foundation;
using namespace Windows::Globalization;
using namespace Windows::Media::Ocr;
using namespace Windows::Graphics::Imaging;
using namespace Windows::Storage::Streams;

extern Ort::MemoryInfo mem_info;

// Tokenizer
sentencepiece::SentencePieceProcessor sp_source;
sentencepiece::SentencePieceProcessor sp_target;

// Global değişkenler
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "opusmt");
Ort::SessionOptions session_options;
Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
Ort::SessionOptions session_opts;
std::unique_ptr<Ort::Session> encoder_session;
std::unique_ptr<Ort::Session> decoder_session;

bool g_fullscreen = true;
std::wstring g_overlayText;


bool InitTranslationEngine()
{
    session_opts.SetIntraOpNumThreads(1);
    session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // SentencePiece modellerini yükle
    sp_source.Load("models/source.spm");  // örnek: opus-mt-en-tr source model
    sp_target.Load("models/target.spm");  // örnek: opus-mt-en-tr target model

    // ONNX modellerini yükle
    encoder_session = std::make_unique<Ort::Session>(env, L"models/encoder_model.onnx", session_options);
    decoder_session = std::make_unique<Ort::Session>(env, L"models/decoder_model.onnx", session_options);


    if (!sp_source.Load("models/source.spm").ok()) return false;
    if (!sp_target.Load("models/target.spm").ok()) return false;
    return true;
}


std::wstring TranslateText(const std::wstring& input) {
    // UTF-16 → UTF-8
    int len = WideCharToMultiByte(CP_UTF8, 0, input.c_str(), -1, nullptr, 0, 0, 0);
    std::string utf8(len, 0);
    WideCharToMultiByte(CP_UTF8, 0, input.c_str(), -1, &utf8[0], len, 0, 0);

    // SentencePiece tokenization
    std::vector<int> input_ids;
    sp_source.Encode(utf8, &input_ids);
    input_ids.insert(input_ids.begin(), 0); // <s>
    input_ids.push_back(2);                // </s>

    int64_t input_length = static_cast<int64_t>(input_ids.size());
    std::vector<int64_t> input_shape = { 1, input_length };

    Ort::Value encoder_input = Ort::Value::CreateTensor<int32_t>(
        mem_info, input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size());

    const char* encoder_input_names[] = { "input_ids" };
    const char* encoder_output_names[] = { "last_hidden_state" };

    auto encoder_outputs = encoder_session->Run(
        Ort::RunOptions{ nullptr },
        encoder_input_names, &encoder_input, 1,
        encoder_output_names, 1);

    Ort::Value& encoder_hidden = encoder_outputs[0];

    std::vector<int32_t> decoder_input_ids = { 0 }; // <s>
    std::vector<int64_t> decoder_shape = { 1, 1 };

    std::vector<int> output_tokens;

    for (int step = 0; step < 128; ++step) {
        Ort::Value decoder_input = Ort::Value::CreateTensor<int32_t>(
            mem_info, decoder_input_ids.data(), decoder_input_ids.size(),
            decoder_shape.data(), decoder_shape.size());

        std::array<Ort::Value, 2> decoder_inputs = {
            std::move(decoder_input),
            encoder_hidden
        };

        const char* decoder_input_names[] = { "input_ids", "encoder_hidden_states" };
        const char* decoder_output_names[] = { "logits" };

        auto decoder_outputs = decoder_session->Run(
            Ort::RunOptions{ nullptr },
            decoder_input_names, decoder_inputs.data(), 2,
            decoder_output_names, 1);

        Ort::Value& logits_tensor = decoder_outputs[0];

        auto shape = logits_tensor.GetTensorTypeAndShapeInfo().GetShape();
        int64_t vocab_size = shape[2];
        float* logits = logits_tensor.GetTensorMutableData<float>();

        float* last_step_logits = logits + (shape[1] - 1) * vocab_size;
        int max_id = static_cast<int>(std::distance(
            last_step_logits,
            std::max_element(last_step_logits, last_step_logits + vocab_size)));

        if (max_id == 2) break;

        output_tokens.push_back(max_id);
        decoder_input_ids.push_back(max_id);
        decoder_shape[1] = decoder_input_ids.size();
    }

    std::string decoded;
    sp_target.Decode(output_tokens, &decoded);

    int wlen = MultiByteToWideChar(CP_UTF8, 0, decoded.c_str(), -1, nullptr, 0);
    std::wstring result(wlen, 0);
    MultiByteToWideChar(CP_UTF8, 0, decoded.c_str(), -1, &result[0], wlen);

    return result;
}



// Dialog
INT_PTR CALLBACK MainDlgProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM)
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
            g_fullscreen = (IsDlgButtonChecked(hDlg, IDC_RADIO_FULLSCREEN) == BST_CHECKED);
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
    HWND hOverlay = CreateWindowExW(
        WS_EX_TOPMOST | WS_EX_LAYERED | WS_EX_TRANSPARENT,
        L"STATIC", L"", WS_POPUP,
        0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN),
        nullptr, nullptr, nullptr, nullptr);
    SetLayeredWindowAttributes(hOverlay, 0, 80, LWA_ALPHA);
    ShowWindow(hOverlay, SW_SHOW);

    POINT start{}, end{};
    MSG msg;
    bool selecting = false;

    while (true)
    {
        if (GetAsyncKeyState(VK_LBUTTON) & 0x8000)
        {
            if (!selecting)
            {
                GetCursorPos(&start);
                selecting = true;
            }
        }
        else if (selecting)
        {
            GetCursorPos(&end);
            break;
        }
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        Sleep(10);
    }

    DestroyWindow(hOverlay);
    return RECT{
        min(start.x, end.x),
        min(start.y, end.y),
        max(start.x, end.x),
        max(start.y, end.y)
    };
}


SoftwareBitmap CaptureScreen(int x, int y, int w, int h)
{
    HDC dcScreen = GetDC(nullptr);
    HDC dcMem = CreateCompatibleDC(dcScreen);
    HBITMAP bmp = CreateCompatibleBitmap(dcScreen, w, h);
    SelectObject(dcMem, bmp);
    BitBlt(dcMem, 0, 0, w, h, dcScreen, x, y, SRCCOPY);

    BITMAPINFOHEADER bi{ sizeof(bi), w, -h, 1, 32, BI_RGB };
    std::vector<uint8_t> pixels(w * h * 4);
    GetDIBits(dcMem, bmp, 0, h, pixels.data(), reinterpret_cast<BITMAPINFO*>(&bi), DIB_RGB_COLORS);

    ReleaseDC(nullptr, dcScreen);
    DeleteDC(dcMem);
    DeleteObject(bmp);

    InMemoryRandomAccessStream stream;
    auto encoder = BitmapEncoder::CreateAsync(BitmapEncoder::BmpEncoderId(), stream).get();
    encoder.SetPixelData(BitmapPixelFormat::Bgra8, BitmapAlphaMode::Ignore, w, h, 96, 96, pixels);
    encoder.FlushAsync().get();

    auto decoder = BitmapDecoder::CreateAsync(stream).get();
    return decoder.GetSoftwareBitmapAsync().get();
}


LRESULT CALLBACK OverlayProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
    if (msg == WM_PAINT)
    {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);
        SetBkMode(hdc, TRANSPARENT);
        SetTextColor(hdc, RGB(255, 255, 0));
        RECT r; GetClientRect(hwnd, &r);
        DrawTextW(hdc, g_overlayText.c_str(), -1, &r, DT_CENTER | DT_VCENTER | DT_WORDBREAK);
        EndPaint(hwnd, &ps);
        return 0;
    }
    if (msg == WM_DESTROY)
    {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, msg, wp, lp);
}

void ShowOverlay(const std::wstring& text)
{
    g_overlayText = text;
    WNDCLASSW wc{ 0 };
    wc.lpfnWndProc = OverlayProc;
    wc.hInstance = GetModuleHandle(nullptr);
    wc.lpszClassName = L"OcrOverlay";
    wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    RegisterClassW(&wc);

    HWND hwnd = CreateWindowExW(
        WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TRANSPARENT,
        wc.lpszClassName, nullptr,
        WS_POPUP, 100, 100, 600, 100,
        nullptr, nullptr, wc.hInstance, nullptr);
    SetLayeredWindowAttributes(hwnd, 0, 220, LWA_ALPHA);
    ShowWindow(hwnd, SW_SHOW);

    std::thread([hwnd]() {
        while (!(GetAsyncKeyState(VK_ESCAPE) & 0x8000)) Sleep(50);
        PostMessage(hwnd, WM_CLOSE, 0, 0);
        }).detach();

    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}


int main()
{
    init_apartment(); // WinRT başlat

    // Ort::Env, SessionOptions ve diğer globaller InitTranslationEngine içinde tanımlı
    if (!InitTranslationEngine())
    {
        std::wcerr << L"[HATA] ONNX + SentencePiece yüklenemedi.\n";
        return -1;
    }

    // Başlangıç: kullanıcıdan tam ekran mı, bölgesel mi seçim al
    INT_PTR dlg = DialogBox(
        GetModuleHandle(nullptr),
        MAKEINTRESOURCE(IDD_MAIN_DIALOG),
        nullptr,
        MainDlgProc);
    if (dlg != IDOK) return 0;

    // Seçilen alana göre ekran bölgesi tanımla
    RECT region{};
    if (g_fullscreen)
    {
        region = {
            0, 0,
            GetSystemMetrics(SM_CXSCREEN),
            GetSystemMetrics(SM_CYSCREEN)
        };
    }
    else
    {
        region = SelectScreenRegion();
        if ((region.right - region.left) < 5 || (region.bottom - region.top) < 5)
        {
            std::wcerr << L"[Uyarı] Seçilen bölge çok küçük." << std::endl;
            return 0;
        }
    }

    // OCR motoru
    OcrEngine engine = OcrEngine::TryCreateFromLanguage(Language(L"en"));
    if (!engine)
    {
        std::wcerr << L"[HATA] OCR motoru başlatılamadı. İngilizce dil paketi yüklü mü?" << std::endl;
        return -1;
    }

    std::wstring lastText;

    // Sürekli izleme döngüsü
    while (true)
    {
        // Görüntü yakalama ve OCR
        auto bmp = CaptureScreen(
            region.left,
            region.top,
            region.right - region.left,
            region.bottom - region.top);

        auto result = engine.RecognizeAsync(bmp).get();
        std::wstring text = result.Text().c_str();

        if (text != lastText && !text.empty())
        {
            lastText = text;

            std::wstring translated = TranslateText(text);

            ShowOverlay(translated);
        }

        // ESC ile döngüden çık
        if (GetAsyncKeyState(VK_ESCAPE) & 0x8000)
            break;

        Sleep(500); // CPU dostu
    }

    return 0;
}

