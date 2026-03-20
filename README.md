# InstaOptima Refactor

Project này tách lại từ script thực nghiệm ban đầu để dễ quản lý, và hiện đã chuyển sang luồng thực nghiệm gần với paper InstaOptima hơn:

- dùng `google/flan-t5-base` làm task model để fine-tune và đánh giá từng instruction
- dùng mô hình OpenAI làm toán tử sinh offspring (`definition/example mutation/crossover`)
- tiến hóa theo vòng lặp `P -> Q -> P ∪ Q -> NSGA-II selection`
- lưu objective của từng cá thể, không reevaluate cha mẹ ở mỗi generation

## Cấu trúc

- `instaoptima/config.py`: cấu hình thí nghiệm
- `instaoptima/llm_client.py`: gọi OpenAI API cho các evolution operators
- `instaoptima/data_loader.py`: tải dữ liệu Hugging Face hoặc local train/validation/test
- `instaoptima/flan_t5_evaluator.py`: fine-tune và suy luận với Flan-T5-base cho từng instruction
- `instaoptima/instruction.py`: model cho instruction
- `instaoptima/evaluator.py`: đánh giá metric và objective
- `instaoptima/perplexity.py`: tính perplexity objective
- `instaoptima/pareto.py`: non-dominated sorting và NSGA-II style selection
- `instaoptima/operators.py`: mutation và crossover
- `instaoptima/population.py`: khởi tạo population ban đầu
- `instaoptima/runner.py`: vòng lặp thực nghiệm chính
- `main.py`: entrypoint chạy chương trình

## Chạy thử

```bash
cp .env.example .env
python3 main.py --config config.yaml
```

File `.env`:

```env
OPENAI_API_KEY=your_key
```

Lưu ý:

- lần chạy đầu sẽ cần tải `google/flan-t5-base` và `roberta-base`
- chi phí tính toán đã tăng đáng kể vì mỗi instruction được fine-tune riêng trên tập train trước khi đo trên tập test
- `config.yaml` mặc định dùng `1000` mẫu train và toàn bộ split đánh giá có nhãn
- với `glue/sst2`, file config mặc định dùng `validation` làm split đánh giá vì split `test` công khai không có label

## Cấu hình YAML

Bạn có thể chỉnh tham số thực nghiệm trong `config.yaml` rồi chạy lại:

```bash
python3 main.py --config config.yaml
```

Để test nhanh end-to-end trước khi chạy cấu hình lớn, có thể dùng preset debug cho ABSA:

```bash
python3 main.py --config config_absa_debug.yaml
```

Các tham số mới quan trọng:

- `task_model_name`: mặc định `google/flan-t5-base`
- `task_model_train_epochs`
- `task_model_learning_rate`
- `task_model_train_batch_size`
- `task_model_eval_batch_size`
- `task_model_max_source_length`
- `task_model_max_target_length`
- `task_model_generation_max_new_tokens`
- `task_model_device`

## Quy trình hiện tại

Code hiện tại chạy theo nhịp:

1. Khởi tạo `M` instruction ban đầu.
2. Evaluate toàn bộ quần thể ban đầu bằng cách fine-tune `Flan-T5-base` trên tập train và chấm trên tập test.
3. Mỗi generation sinh đúng `M` offspring từ toàn bộ quần thể cha mẹ.
4. Mỗi offspring được evaluate ngay sau khi sinh.
5. Gộp `P ∪ Q` rồi chọn lại `M` cá thể bằng non-dominated sort và crowding distance.
6. Xuất Pareto front cuối cùng.

Để chạy đúng `Laptop14` hoặc `Restaurant14`, bạn nên đổi `dataset_source: local` và cung cấp:

- `local_train_path`
- `local_validation_path`
- `local_test_path`
- `task_type: absa`
- `aspect_field`
- `label_space`

Nếu file local chưa tồn tại và `auto_download_local_dataset: true`, loader sẽ tự tải bản ABSA tương ứng từ Hugging Face, tách validation nếu cần, rồi ghi ra các file `local_train_path`, `local_validation_path`, `local_test_path`.
