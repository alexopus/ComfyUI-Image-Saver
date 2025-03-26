import requests

class CivitaiHashFetcher:
    """
    A ComfyUI custom node that fetches the AutoV3 hash of a model from Civitai
    based on the provided username and model name.
    """

    def __init__(self):
        self.last_username = None
        self.last_model_name = None
        self.last_version = None
        self.last_hash = None  # Store the last fetched hash

    RETURN_TYPES = ("STRING",)  # The node outputs a string (AutoV3 hash)
    FUNCTION = "get_autov3_hash"
    CATEGORY = "CivitaiAPI"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "username": ("STRING", {"default": "", "multiline": False}),
                "model_name": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "version": ("STRING", {"default": "", "multiline": False, "tooltip": "Specify version keyword to fetch a particular model version (optional)"}),
            }
        }

    def get_autov3_hash(self, username, model_name, version=""):
        """
        Fetches the latest model version from Civitai and extracts its AutoV3 hash.
        Uses caching to avoid redundant API calls.
        """
        # Check if inputs are the same as last time
        if (self.last_username is not None and self.last_model_name is not None and self.last_version is not None and 
            username == self.last_username and model_name == self.last_model_name and version == self.last_version):
            return self.last_hash

        base_url = "https://civitai.com/api/v1/models"
        params = {
            "username": username,
            "query": model_name,
            "limit": 1  # Get the most relevant model
        }

        try:
            # Fetch models by username and model name
            response = requests.get(base_url, params=params)
            if response.status_code != 200:
                return (f"Error: API request failed with status {response.status_code}",)

            data = response.json()
            items = data.get("items", [])
            if not items:
                return (f"No models found for user '{username}' with name '{model_name}'",)

            # Take the first model from the search results
            model = items[0]
            model_versions = model.get("modelVersions", [])
            if not model_versions:
                return ("No model versions found.",)

            # If a version keyword is provided, search for a model version whose name contains it (case-insensitive).
            chosen_version = None
            if version:
                for v in model_versions:
                    if version.lower() in v.get("name", "").lower():
                        chosen_version = v
                        break
            # If no version is provided or no match was found, use the first (latest) version.
            if chosen_version is None:
                chosen_version = model_versions[0]
            version_id = chosen_version.get("id")

            # Fetch detailed version info
            version_url = f"https://civitai.com/api/v1/model-versions/{version_id}"
            version_response = requests.get(version_url)
            if version_response.status_code != 200:
                return (f"Error: Version API request failed with status {version_response.status_code}",)

            version_data = version_response.json()

            # Extract the AutoV3 hash from the model version files
            for file_info in version_data.get("files", []):
                autov3_hash = file_info.get("hashes", {}).get("AutoV3")
                if autov3_hash:
                    # Cache the result before returning
                    self.last_username = username
                    self.last_model_name = model_name
                    self.last_version = version  # Store version to track changes
                    self.cached_hash = autov3_hash
                    return (autov3_hash,)  # Return the first found hash

            return ("No AutoV3 hash found in version files.",)

        except Exception as e:
            return (f"Error: {e}",)
