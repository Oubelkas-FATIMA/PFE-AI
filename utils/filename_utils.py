def clean_filename(name):
    '''
    Clean a filename by removing invalid characters.
    
    :param name: Input filename
    :return: Cleaned filename
    '''
    return "".join(c for c in name if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()
