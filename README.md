
# Hệ thống nhận diện món ăn ở căn tin

## Tổng quan về dự án

Dự án này đề xuất một hệ thống trí tuệ nhân tạo có khả năng tự động nhận diện các món ăn được lựa chọn trên khay tại quầy thanh toán của căn tin thông qua phân tích hình ảnh, từ đó xác định và ánh xạ giá tiền tương ứng từ cơ sở dữ liệu thực đơn. Mục tiêu của dự án là giảm thiểu thời gian giao dịch thanh toán, nâng cao độ chính xác trong việc nhận diện món ăn và tính toán chi phí, cung cấp một phương thức thanh toán hiện đại và tiện lợi, cũng như tạo cơ sở dữ liệu về thông tin bán hàng.

## Hướng dẫn cài đặt

1.  Môi trường lập trình:
    * Ngôn ngữ lập trình: Python 
    * Thư viện và khuôn khổ học sâu: PyTorch, TensorFlow 
    * Các thư viện hỗ trợ:  matplotlib (cho cả YOLO và CNN), Ultralytics YOLOv8, OpenCV (xử lý ảnh), Pandas (xử lý dữ liệu), YAML (cấu hình), tensorflow, numpy

2.  Phần cứng:
    * Máy tính (PC) với GPU để tăng tốc quá trình huấn luyện và suy luận 
    * Camera 
    * Arduino
    * nút nhấn

3.  **Các bước cài đặt (chi tiết):**
    * Cài đặt Python và các thư viện cần thiết (PyTorch, TensorFlow, OpenCV,keras)
    * Cài đặt Ultralytics YOLOv8.
    * Kết nối camera với máy tính.
    * Cài đặt và cấu hình Arduino 

 Hướng dẫn sử dụng

1.  Thu thập ảnh: Đặt khay thức ăn dưới camera. 
2.  Kích hoạt hệ thống: Nhấn nút để chụp ảnh khay thức ăn. 
3.  Xử lý ảnh: Hệ thống sẽ tự động nhận diện các món ăn và tính toán tổng tiền. 
4.  Hiển thị kết quả: Thông tin về các món ăn và tổng giá tiền sẽ hiển thị trên màn hình máy tính. 

 Các phần phụ thuộc

* Python
* PyTorch, TensorFlow
* matplotlib
* Ultralytics YOLOv8
* OpenCV
* Pandas
* YAML
* GPU 
* Camera
* Arduino 
