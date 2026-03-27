import os
import importlib.util


class CodeEvalImportHook:
    def find_spec(self, fullname, path, target=None):
        # Intercept execute imports (working logic)
        if 'execute' in fullname and ('code_eval' in fullname or 'evaluate' in str(path)):
            # Redirect to custom execute.py
            custom_path = '/home/gensyn/persistent/DetGatingMoE-exp/utils/execute.py'
            if os.path.exists(custom_path):
                spec = importlib.util.spec_from_file_location(fullname, custom_path)
                print(f"Redirecting {fullname} to {custom_path}")
                return spec
        return None