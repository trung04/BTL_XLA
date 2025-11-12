  // Hiển thị ảnh preview
        function previewImage(event) {
            const file = event.target.files[0];
            if (!file) return;
            const img = document.getElementById('preview');
            img.src = URL.createObjectURL(file);
            img.hidden = false;
        }

        // Reset form
        function resetForm() {
            document.getElementById('imageInput').value = '';
            document.getElementById('preview').hidden = true;
            document.getElementById('shapeName').textContent = '—';
            document.getElementById('confidenceScore').textContent = '';
            document.getElementById('resultLabel').textContent = 'Chưa có kết quả';
            document.getElementById('resultBox').style.border = '2px dashed #bfdbfe';
        }

        // Gọi model dự đoán
        async function predict() {
            const imgInput = document.getElementById('imageInput');
            if (!imgInput.files.length) {
                alert('⚠️ Vui lòng chọn ảnh trước!');
                return;
            }

            const formData = new FormData();
            formData.append('image', imgInput.files[0]);

            try {
                const res = await fetch('http://127.0.0.1:5000/recognize2', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();

                document.getElementById('shapeName').textContent =
                    data.shape === 'circle' ? 'Hình tròn 🟢' : 'Hình chữ nhật ⬜';
                document.getElementById('confidenceScore').textContent =
                    `Độ tin cậy: ${(data.confidence * 100).toFixed(2)}%`;
                document.getElementById('resultLabel').textContent = '✅ Kết quả dự đoán:';
                document.getElementById('resultBox').style.border = '2px solid #3b82f6';
            } catch (err) {
                console.error(err);
                alert('❌ Lỗi khi gửi ảnh đến server!');
            }
        }