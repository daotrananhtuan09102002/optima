# InstaOptima Refactor

Project này tách lại từ script thực nghiệm ban đầu để dễ quản lý, đồng thời đang được nâng dần theo setup của paper InstaOptima.

## Cấu trúc

- `instaoptima/config.py`: cấu hình thí nghiệm
- `instaoptima/llm_client.py`: gọi OpenAI API
- `instaoptima/data_loader.py`: tải dữ liệu Hugging Face hoặc local train/validation/test
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

## Cấu hình YAML

Bạn có thể chỉnh tham số thực nghiệm trong `config.yaml` rồi chạy lại:

```bash
python3 main.py --config config.yaml
```

Để test nhanh end-to-end trước khi chạy cấu hình lớn, có thể dùng preset debug cho ABSA:

```bash
python3 main.py --config config_absa_debug.yaml
```

## Gần paper hơn

Code hiện đã có:

- objective vector gồm performance, length, perplexity
- 4 evolution operators
- multi-run summary
- selection kiểu NSGA-II đơn giản
- hỗ trợ dữ liệu local cho bài toán ABSA như Laptop14 và Restaurant14

Để chạy đúng `Laptop14` hoặc `Restaurant14`, bạn nên đổi `dataset_source: local` và cung cấp:

- `local_train_path`
- `local_validation_path`
- `local_test_path`
- `task_type: absa`
- `aspect_field`
- `label_space`

Nếu file local chưa tồn tại và `auto_download_local_dataset: true`, loader sẽ tự tải bản ABSA tương ứng từ Hugging Face, tách validation nếu cần, rồi ghi ra các file `local_train_path`, `local_validation_path`, `local_test_path`.
