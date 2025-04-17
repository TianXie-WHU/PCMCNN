import xlrd
import torch


def read_xlrd(excelFile, sheet_name):
    """
    Read data from specified Excel sheet and return as list of rows

    Args:
        excelFile (str): Path to Excel file (.xls or .xlsx)
        sheet_name (str): Name of worksheet to read

    Returns:
        list: 2D list containing row data (excluding header row)

    Raises:
        xlrd.XLRDError: If sheet name not found or file format invalid
    """
    # Open Excel workbook using xlrd library
    data = xlrd.open_workbook(excelFile)

    # Access specified worksheet by name
    table = data.sheet_by_name(sheet_name)

    # Initialize empty list to store processed data
    dataFile = []

    # Iterate through all rows in worksheet
    for rowNum in range(table.nrows):
        # Skip header row (row index 0)
        if rowNum > 0:
            # Append row values as sublist to dataFile
            dataFile.append(table.row_values(rowNum))

    # Return 2D list containing all data rows
    return dataFile
