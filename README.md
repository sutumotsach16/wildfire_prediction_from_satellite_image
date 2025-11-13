# Xây dựng mô hình dự đoán cháy rừng từ ảnh vệ tinh

Mô hình này sử dụng mạng nơ-ron tích chập (CNN) để phát hiện cháy rừng từ ảnh vệ tinh. Dự án bao gồm các bước tiền xử lý dữ liệu, xây dựng mô hình, huấn luyện và đánh giá.

## 📚 Mục lục

* [Giới thiệu](#giới-thiệu)
* [Cài đặt](#cài-đặt)
* [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
* [Cách sử dụng](#cách-sử-dụng)
* [Đóng góp](#đóng-góp)
* [Kết quả](#kết-quả)

---

## Giới thiệu
Dự án này nhằm phát triển một mô hình học sâu có khả năng nhận diện cháy rừng từ ảnh vệ tinh, giúp cảnh báo sớm và giảm thiểu thiệt hại do chá

## Cài đặt
Để bắt đầu, bạn cần làm theo các bước sau:
1. Cài đặt Python 3.8 trở lên từ [python.org](https://www.python.org/downloads/).
2. Tạo một môi trường ảo (virtual environment) để quản lý các gói thư viện:
   ```bash
   python -m venv venv
   ```
3. Kích hoạt môi trường ảo:
   - Trên Windows:
     ```bash
     venv\Scripts\activate
     ```
    - Trên macOS/Linux:
      ```bash
      source venv/bin/activate
      ```
4. Cài đặt các thư viện cần thiết từ tệp `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

### Yêu cầu hệ thống
* Python 3.8 trở lên
* RAM tối thiểu 4GB

## Cách sử dụng
Chạy file `python prediction.py` để khởi động ứng dụng. Tùy thuộc vào công cụ bạn sử dụng, bạn có thể chạy file theo các cách sau:
- Trên terminal hoặc command prompt:
  ```bash
  python prediction.py
  ```
- Trong môi trường Jupyter Notebook, bạn có thể sử dụng:
  ```python 
  !python prediction.py
  ```  
- Trong IDE như PyCharm hoặc VSCode, bạn có thể mở file `prediction.py` và chạy trực tiếp từ giao diện IDE.

## Kết quả

| Ảnh dự đoán 1 | Ảnh dự đoán 2 |
|:-------:|:-----------:|
| ![Ảnh dự đoán 1](./report/pic/Screenshot%202025-11-13%20231919.png) | ![Ảnh dự đoán 2](./report/pic/Screenshot%202025-11-13%20232003.png) |