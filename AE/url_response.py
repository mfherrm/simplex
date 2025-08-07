from AE.extension_data_response import ExtensionDataResponse


class UrlResponse(ExtensionDataResponse):
    def __init__(self, url: str):
        content_type = 'text/uri-list'
        super().__init__(content_type)

        self._url = url

    def get_response(self, arg1, arg2):
        return self._create_response(self._url)