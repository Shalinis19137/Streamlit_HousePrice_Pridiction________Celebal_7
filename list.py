#list = [3,5,9,2,1]
#s_li = sorted(list)  # sort list in Assending order
#s_list = sorted(list,reverse = True ) # sort list in Desending order
#print(s_li)

#list.sort()   # other way to sort list in Asscending order 
#list.sort(reverse = True)   # other way to sort list in Descending order


list_1 = [-3,-5,-9,2,1]
s_li_1 = sorted(list_1, key=abs)  # sort list in Descending order
s_list_1 = sorted(list_1,reverse = True ) # sort list in Asscending order
print(s_li_1)
