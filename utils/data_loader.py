import xlrd
import torch

def read_xlrd(excelFile, sheet_name):
    data = xlrd.open_workbook(excelFile)
    table = data.sheet_by_name(sheet_name)
    dataFile = []

    for rowNum in range(table.nrows):
        if rowNum > 0:
            dataFile.append(table.row_values(rowNum))

    return dataFile

