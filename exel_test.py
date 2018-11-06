import xlrd

book = xlrd.open_workbook('data_book.xlsx')
sheet = book.sheet_by_index(0)
print (sheet.cell(0,0))
print (sheet.cell(1,0))
print (5*0)