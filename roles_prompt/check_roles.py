import json
import os
from pathlib import Path

def check_roles(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            roles = data.get('roles', {})
            return roles
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def is_subset_structure(subset, superset):
    """Check if all values in the subset structure are contained in the superset structure."""
    for key, value in subset.items():
        if key not in superset or not set(value).issubset(set(superset[key])):
            return False
    return True

def main():
    base_dir = Path('processed/programdev/chatdev')
    all_roles = []
    seen_roles = set()
    role_files = {}
    empty_role_files = []
    
    # Walk through all directories under AG2
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = Path(root) / file
                roles = check_roles(file_path)
                if roles is not None:
                    if not roles:  # Check if roles is empty
                        empty_role_files.append(file_path)
                        continue
                    # Convert roles dict to a string for hashing
                    roles_str = json.dumps(roles, sort_keys=True)
                    seen_roles.add(roles_str)
                    if roles_str not in role_files:
                        role_files[roles_str] = []
                    role_files[roles_str].append(str(file_path))
    
    # Handle empty role files
    if empty_role_files:
        print(f"\nFound {len(empty_role_files)} files with empty role structures:")
        for file_path in empty_role_files:
            print(f"Deleting: {file_path}")
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

    # Filter out redundant role structures
    filtered_roles = []
    for roles_str in seen_roles:
        roles = json.loads(roles_str)
        if not any(is_subset_structure(roles, json.loads(other_roles_str)) for other_roles_str in seen_roles if roles_str != other_roles_str):
            filtered_roles.append(roles_str)

    # Print results
    print(f"Total files checked: {sum(len(files) for files in role_files.values())}")
    print(f"Found {len(filtered_roles)} different role structures:")
    
    for roles_str in filtered_roles:
        roles = json.loads(roles_str)
        print("\nRole values:")
        print(json.dumps(list(roles.values()), indent=2))
        print(f"Files with this structure ({len(role_files[roles_str])}):")
        for file_path in role_files[roles_str][:3]:  # Show only first 3 files
            print(f"- {file_path}")
        if len(role_files[roles_str]) > 3:
            print(f"... and {len(role_files[roles_str]) - 3} more files")
        print("-" * 50)

if __name__ == "__main__":
    main()
