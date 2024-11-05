import os

def find_entries(base_path, entries):
    found_entries = set()
    
    for entry in entries:
        # Tạo đường dẫn đầy đủ cho entry
        for root, dirs, files in os.walk(base_path):
            # Kiểm tra nếu entry là thư mục
            if entry in dirs:
                found_entries.add(os.path.join(root, entry))
            # Kiểm tra nếu entry là file
            if entry in files:
                found_entries.add(os.path.join(root, entry))

    return found_entries

def add_to_gitignore(entries):
    gitignore_path = '.gitignore'

    # Nếu file .gitignore không tồn tại, tạo mới
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, 'w') as f:
            f.write('# .gitignore\n')

    # Tìm kiếm các entry trong cây thư mục
    base_path = os.getcwd()  # Sử dụng thư mục hiện tại
    found_entries = find_entries(base_path, entries)

    # Mở file .gitignore để thêm các entry
    with open(gitignore_path, 'a') as f:
        for entry in found_entries:
            # Kiểm tra xem entry đã có trong .gitignore chưa
            if entry not in open(gitignore_path).read():
                f.write(entry + '\n')
                print(f'Added: {entry}')
            else:
                print(f'Already exists: {entry}')

# Danh sách các file và folder cần thêm vào .gitignore
entries = [
    '.venv',                # Thư mục ảo (virtual environment)
    'python=3.10.12',      # Tên file (điều này cần phải là một file thực tế nếu muốn)
]

add_to_gitignore(entries)
