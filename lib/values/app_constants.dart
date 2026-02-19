class AppConstants {
  const AppConstants._();

  static const instance = AppConstants._();

  // OLD - SSD MobileNet (comment out or delete)
  // static const String ssdMobileNetV1 = 'assets/model/ssd_mobilenet_v1.tflite';
  // static const String ssdMobileNetV1LabelPath = 'assets/label/labels.txt';
  // static const int ssdCompatibleImageHeight = 300;
  // static const int ssdCompatibleImageWidth = 300;

  // NEW - YOLOv8n
  static const String ssdMobileNetV1 = 'assets/yolov8n_float32.tflite';
  static const String ssdMobileNetV1LabelPath = 'assets/coco_labels.txt';

  static const int ssdCompatibleImageHeight = 640;
  static const int ssdCompatibleImageWidth = 640;
}