import laspy
import os
import shutil

def cut(infile,outfile):
    lasfile = laspy.read(infile)
    # 控制 x,z 范围
    mask = (lasfile.x < 200) & (lasfile.z > -20)
    las = lasfile[mask]
    las.write(outfile)
    
def process_all_las_files(input_dir,output_dir):
    if not os.path.exists(output_dir):
        print('输出的文件夹不存在')
        return
    
    # 遍历输入目录中的所有las文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.las'):
            # 分割文件名与扩展名
            base_name,ext = os.path.splitext(filename)
            # 构建新的文件名
            new_base_name = f"{base_name}_pp"
            new_filename = f"{new_base_name}{ext}"
            
            infile_path = os.path.join(input_dir,filename)
            outfile_path = os.path.join(output_dir,new_filename)
            
            # 调用cut函数
            cut(infile_path,outfile_path)
            print(f'Output file has been written to: {outfile_path}') 
    
    
if __name__ == '__main__':
    input_dir = r'data\origin_data'  
    output_dir = r'data\pp_data'  
    process_all_las_files(input_dir, output_dir)
    

    