import random
from PIL import Image, ImageDraw,ImageFont


class gen_code():
    '''
    生成验证码
    '''
    def gen_code(self):
       # return chr(random.randint(65,90)) # 随机字母
       return str(random.randint(0,9)) # 随机字母
    def bg_color(self):
        return (random.randint(0,150),random.randint(50,160),random.randint(150,250)) #随机背景
    def fr_color(self):
        return (random.randint(150,250),random.randint(150,250),random.randint(0,160))	#随机前景
    def gen_pic(self, train_num, test_num):
        all = int(train_num) + int(test_num)
        print(f"共需生成{all}张验证码")
        for num in range(int(all)):
            w, h = 120,60
            trage = ''
            img = Image.new(size=(w,h), mode='RGB', color=(255,255,255)) # Image.new 创建新图片
            draw = ImageDraw.Draw(img)	# 在图片img上画字需要调用包
            font = ImageFont.truetype(font="C:/Windows/Fonts/Arial.ttf",size=30) # 设置字体，需要导包
            for y in range(h):
                for x in range(w):
                    draw.point((x,y), fill=self.bg_color()) # draw.point对img上每个像素赋值
            for i in range(4):
                code = self.gen_code()
                trage = trage + code
                draw.text((30*i + 10,20),text=code,fill=self.fr_color(),font=font) # 在图片上写字 fill为字体颜色text为所写类容
            # img.show()
            # print(trage)
            if num < int(train_num):
                img.save(rf"D:\data\RNN\img\train\{trage}.jpg") # 保存图片
            else:
                img.save(rf"D:\data\RNN\img\test\{trage}.jpg") # 保存图片
            print(f"已生成{num+1}张图片")
            num += 1
        print("验证码生成完毕")

if __name__ == '__main__':
    train_num = input("请输入用于训练的验证码数量")
    test_num = input("请输入用于测试的验证码数量")
    code = gen_code()
    code.gen_pic(train_num, test_num)