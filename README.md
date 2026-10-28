### Giới thiệu
Đề tài: Xây dựng hệ thống nhận diện nội dung độc hại trong bình luận tiếng Việt 
Mô hình:
- baseline: TF-IDF + Support Vector Machine (SVM)
- PhoBERT
- VisoBERT

### Cài đặt
- B1: tải source code từ link github sau: https://github.com/Phuonguyen06-pu/Sourcecode_NLP.git
- B2: giải nén, mở source code trong vs code
- B3: mở terminal, chạy `pip install -r requirements.txt` để cài các thư viện
- B4: tải các mô hình đã huấn luyện từ Hugging Face về `saved_models/`, bằng cách chạy `python download_models.py`
- B5: chạy `python ui/api.py` rồi copy đường dẫn lên web để chạy

Note: vì file trọng số mô hình quá nặng không push được lên github nên nhóm sẽ để lên hugging face, khi chạy sẽ tải mô hình về
link trọng số mô hình: https://huggingface.co/Uyen1

### Nguồn dữ liệu
https://huggingface.co/datasets/uitnlp/vihsd
