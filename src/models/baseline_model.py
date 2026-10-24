'''Gói mô hình baseline đã huấn luyện để dự đoán thẳng từ chuỗi văn bản'''
 
 
class TextSVM:
 
    def __init__(self, vector, svm, calibrator):
        self.vector = vector
        self.svm = svm
        self.calibrator = calibrator
        self.classes_ = calibrator.classes_  # các lớp ứng với từng cột của predict_proba
 
    def predict(self, texts):
        '''nhãn dự đoán, lấy trực tiếp từ SVM'''
        return self.svm.predict(self.vector.transform(texts))
 
    def predict_proba(self, texts):
        '''xác suất từng lớp (dùng làm độ tin cậy)'''
        scores = self.svm.decision_function(self.vector.transform(texts))
        return self.calibrator.predict_proba(scores)
