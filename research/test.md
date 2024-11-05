4. CosFace và ArcFace
    Ý tưởng: Thêm margin góc vào softmax để tạo khoảng cách giữa các lớp.
        CosFace trừ margin vào cosine của góc giữa đặc trưng và trọng số lớp.
        ArcFace đưa margin vào trong hàm cosine, tạo margin dạng cung (angular margin) để cải thiện tính phân biệt.
    Cải tiến: CosFace và ArcFace tăng độ chính xác nhận diện, tối ưu hóa phân tách giữa các lớp bằng cách trực tiếp thao tác trên góc.

    Giới hạn: Đòi hỏi điều chỉnh các siêu tham số nhạy cảm, dễ làm mất ổn định khi huấn luyện.

5. SV-Softmax
    Ý tưởng: SV-Softmax chọn các mẫu âm khó (hard negative) và đẩy nó ra xa trung tâm các lớp positive, giúp cải thiện sự phân tách nội lớp.

    Cải tiến: Đưa ra chiến lược hard negative mining, đẩy mạnh sự gắn kết của các mẫu trong lớp dương bằng cách xử lý các mẫu âm khó.

6. Ring Loss
    Ý tưởng: Giữ độ dài của các embedding không đổi ở giá trị 𝑅, giúp embedding có cùng độ lớn và tăng cường tính ổn định.

    Cải tiến: Hỗ trợ các hàm mất mát khác duy trì độ dài embedding, giảm thiểu ảnh hưởng của nhiễu trong không gian đặc trưng.

7. MagFace
    Ý tưởng: Điều chỉnh margin theo độ lớn của đặc trưng khuôn mặt theo chất lượng mẫu. Mẫu dễ sẽ có độ lớn cao và nằm gần trung tâm, mẫu khó hoặc nhiễu có độ lớn nhỏ và cách xa.
    
    Cải tiến: Kết hợp cả margin và độ lớn, giúp tối ưu hóa phân bố của các mẫu dễ và khó trong không gian đặc trưng, đồng thời tăng tính chống nhiễu.

8. AdaFace
    Ý tưởng: Điều chỉnh gradient của các mẫu khó dựa trên chất lượng của ảnh. Khi chất lượng cao, mẫu khó được nhấn mạnh; khi chất lượng thấp, mức độ ảnh hưởng của mẫu khó giảm.

    Cải tiến: Tự động điều chỉnh margin theo chất lượng dữ liệu, cải thiện tính ổn định của mô hình khi gặp dữ liệu đa dạng về chất lượng.

9. Sub-center ArcFace
    Ý tưởng: Chia các mẫu của một danh tính thành nhiều sub-center, với một sub-center chứa các mẫu sạch và các sub-center còn lại chứa mẫu khó hoặc nhiễu.

    Cải tiến: Giảm áp lực ràng buộc nội lớp, cải thiện khả năng phân loại khi có dữ liệu nhiễu bằng cách xử lý dữ liệu theo từng sub-class.

10. CurricularFace
    Ý tưởng: Áp dụng curriculum learning để học từ các mẫu dễ ở giai đoạn đầu và các mẫu khó ở giai đoạn sau của quá trình huấn luyện.

    Cải tiến: Cải thiện độ chính xác và tính phân biệt bằng cách tập trung vào mẫu dễ trước khi chuyển sang mẫu khó, cập nhật trọng số động qua Exponential Moving Average (EMA) để tối ưu hóa quá trình huấn luyện.

11. NPCface
    Ý tưởng: Nhấn mạnh các mẫu khó cả về dương và âm thông qua collaborative margin để xử lý các tập dữ liệu lớn, nơi các mẫu hard positive và hard negative thường xuất hiện cùng nhau.

    Cải tiến: Tăng khả năng phân biệt với các tập dữ liệu lớn, giúp mô hình tập trung vào các mẫu khó một cách toàn diện hơn.

12. UniformFace
    Ý tưởng: Phân bố đồng đều các lớp trong không gian đặc trưng trên một hypersphere, tối ưu hóa không gian đặc trưng bằng cách giữ khoảng cách tối thiểu giữa các lớp.
    
    Cải tiến: Tối đa hóa khả năng khai thác không gian đặc trưng, giảm thiểu hiện tượng chồng chéo giữa các lớp và tăng cường khả năng phân biệt.
