import os,urllib.request,time

class dailog(object):
    def __init__(self,num):
        super(dailog, self).__init__()
        self.num = num
    def check(self):
        num=self.num
        num1=0
        url=['https://blog.csdn.net/m0_46345373/article/details/119837536']
        res=[]
        while True:
            for x in url:
                try:
                    s=urllib.request.urlopen(x)
                    res.append(s)
                #使用urllib2模块的urlopen函数来打开网页，
                # 并将结果添加到res列表中。这段代码的含义是打开这3个网页，
                # 并将它们的响应对象保存到一个列表中。

                except:

                    res.append(None)

                if not any(res):

                    print ("拨号中") #dai:宽带连接名称,gb39301:账号,111111:密码

                    os.popen("rasphone.exe")
                    num1+=1

                else:

                    print ("network is ok" )

                time.sleep(1)
                if num1 == 10:
                    print ("try reconect 10 ago ,error")
                    break
if __name__ == '__main__':
    p=dailog(60)
    p.check()

