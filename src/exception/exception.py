import sys
from src.logger.logger import logging

def error_message_detail(error, error_detail: sys):
    """
    Extrae los detalles del error: nombre del archivo, número de línea y mensaje de error.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error en el script de Python: [{0}] línea número: [{1}] mensaje de error: [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message