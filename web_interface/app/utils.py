"""Auxiliary functions for implementing the business logic of the application.

Functions
---------
change_image_name
    Create a standardized filename.
"""


def change_image_name(filename: str) -> str:
    """Create a standardized filename.

    Based on the passed file name, a new file name is generated
    and returned, consisting of the standard name and extension
    corresponding to the passed file.
    """

    if filename.find('.') != -1:
        _, extension = filename.rsplit('.', 1)
        new_filename = f'uploaded_image.{extension.lower()}'
        return new_filename
    else:
        raise ValueError('No extension in this filename')
