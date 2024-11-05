import os
import fnmatch

def add_to_gitignore(root_dir, directory_name_patterns, file_name_patterns):
    """
    Thêm các đường dẫn của thư mục và file khớp với mẫu vào file .gitignore.
    
    Parameters:
    - root_dir: Đường dẫn thư mục gốc của project
    - directory_name_patterns: Danh sách các mẫu tên thư mục cần bỏ qua
    - file_name_patterns: Danh sách các mẫu tên file cần bỏ qua
    """

    # Đường dẫn file .gitignore
    gitignore_path = os.path.join(root_dir, '.gitignore')

    # Đọc nội dung hiện tại của .gitignore để tránh trùng lặp
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as gitignore:
            existing_ignores = set(line.strip() for line in gitignore if line.strip())
    else:
        existing_ignores = set()

    # Duyệt qua tất cả các thư mục và file trong root_dir
    with open(gitignore_path, 'a') as gitignore:
        for foldername, subfolders, filenames in os.walk(root_dir):
            # Thêm các thư mục có tên khớp với bất kỳ mẫu nào trong danh sách
            for subfolder in subfolders:
                if any(fnmatch.fnmatch(subfolder, pattern) for pattern in directory_name_patterns):
                    rel_path = os.path.relpath(os.path.join(foldername, subfolder), root_dir)
                    ignore_entry = f'/{rel_path}'
                    if ignore_entry not in existing_ignores:
                        gitignore.write(ignore_entry + '\n')
                        existing_ignores.add(ignore_entry)

            # Thêm các file có tên khớp với bất kỳ mẫu nào trong danh sách
            for filename in filenames:
                if any(fnmatch.fnmatch(filename, pattern) for pattern in file_name_patterns):
                    rel_path = os.path.relpath(os.path.join(foldername, filename), root_dir)
                    ignore_entry = f'/{rel_path}'
                    if ignore_entry not in existing_ignores:
                        gitignore.write(ignore_entry + '\n')
                        existing_ignores.add(ignore_entry)

    print("Đã thêm các thư mục và file theo mẫu vào .gitignore.")

# Thư mục root của project
root_directory = os.path.dirname(os.path.abspath(__file__))

# Danh sách các mẫu tên thư mục và file cần bỏ qua
directory_patterns = ["__pycache__"]  # Mẫu tên thư mục
file_patterns = ["*.keras", "*.csv"]  # Mẫu tên file

# Gọi hàm để thêm vào .gitignore
add_to_gitignore(root_directory, directory_patterns, file_patterns)
