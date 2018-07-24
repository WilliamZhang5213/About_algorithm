# -- coding: utf-8 --
import os, math
import numpy as np
import time

    
def selectionSort(inputs):
    """
        选择排序：在依次比较的过程中交换位置。
    """
    arr = inputs[:]
    N = len(arr)
    
    start = time.time()
    for i in range(0, N):
        small = i
        for j in range(i + 1, N):
            if arr[j] < arr[small]:
                small = j

        arr[i], arr[small] = arr[small], arr[i]
    
    print ("Time:{}".format(time.time() - start))
            
    return arr

def insertionSort(inputs):
    """
        插入排序：将一个数插入到有序的数组中，逆序比较只需要至少向后移一位，而顺序比较择需要移N-1位。
    """
    
    arr = inputs[:]
    N = len(arr)

    # print arr
    start = time.time()
    for i in range(1, N):
        j = i - 1
        key = arr[i]
        while j >= 0 and key < arr[j]:
            arr[j + 1], arr[j] = arr[j], key
            j -= 1
    print ("Time:{}".format(time.time() - start))

    return arr

def shell(inputs):

    arr = inputs[:]
    N = len(arr)

    start = time.time()
    dk = N/2
    while dk > 0:
        for i in range(dk, N):
            j = i
            while j >= dk and arr[j] < arr[j - dk]:
                arr[j - dk], arr[j] = arr[j], arr[j - dk]
                j -= dk
        dk /= 2
    print ("Time:{}".format(time.time() - start))
    
    return arr

def quicksort(inputs):

    arr = inputs[:]
    N = len(arr)


    if N < 2:
        return arr
    else:
        mid = arr[0]
        small = [i for i in arr[1:] if i <= mid]
        large = [i for i in arr[1:] if i > mid]
        return quicksort(small) + [mid] + quicksort(large)
    

def bubbleSort(nums):
    for i in range(len(nums)-1):    # 这个循环负责设置冒泡排序进行的次数
        for j in range(len(nums)-i-1):  # ｊ为列表下标
            if nums[j] > nums[j+1]:
                nums[j], nums[j+1] = nums[j+1], nums[j]
    return nums



if __name__ == "__main__":
    """
        arr = self.inputs 赋值的是地址， 一旦arr地址中的对象改变，self.inputs的值也会改变。
        arr = self.inputs[:] 赋值的是数组的值，不改变arr指向的地址。
    """
    # inputs = [2,8,0,9,4,7,1]

    # print ("Input list:{}".format(inputs))

    # print ("Selection sort:{}".format(selectionSort(inputs)))
    # print ("Insertion sort:{}".format(insertionSort(inputs)))
    # print ("Shell sort:{}".format(shell(inputs)))
    # print ("Quick sort:{}".format(quicksort(inputs)))
    
    nums = [5,2,45,6,8,2,1]

    print(bubbleSort(nums))
                

