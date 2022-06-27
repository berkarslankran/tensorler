
#Mevcut Veriden Tensör Oluşturma
import torch
scaler = 12
array = [1,2]
matrix = [[2.5,2.0,3.0],[4.2,-77,3.21]]

s_tensor = torch.tensor(scaler, dtype=torch.int32)
a_tensor = torch.tensor(array, dtype=torch.float32)
m_tensor = torch.tensor(matrix)

print(f"Ölçekleyiciler  \n{s_tensor}\n")
print(f"1D dizisi \n{a_tensor}\n")
print(f"2D matris \n{m_tensor}\n")

#Sabit değerler kullanma
import torch
rand_uni = torch.rand((3,4)) # tekdüze rastgele değerlere sahip matris
rand_nor = torch.randn(2,3) # normal dağılımdan rastgele değerlere
                            # sahip matris
all_ones = torch.ones(6) # Bir ile dolu 1D dizisi
all_zeros = torch.zeros([2,3,2]) # 3 sıralı sıfır tensörü
all_six = torch.full((2,2), 6) # tüm değerleri 6'ya eşit olan matris
regular = torch.arange(1,2,0.2) # 0,2 aralıklı [1,2) aralığında 1B değer
                                # dizisi

print(f"Tekdüze dağılımdan rastgele \n{rand_uni}\n")
print(f"Normal dağılımdan rastgele  \n{rand_nor}\n")
print(f"Birler  \n{all_ones}\n")
print(f"Sıfırlar  \n{all_zeros}\n")
print(f"Sabit  \n{all_six}\n")
print(f"Bir dizi değer  \n{regular}\n")

#Diğer tensörleri kullanma
import torch
rand_nor = torch.randn(3,5, dtype=torch.float32) # rastgele tensör
all_sevens = torch.zeros_like(rand_nor) # aynı şekil ve veri türü
all_ones = torch.ones_like(rand_nor, dtype=torch.float64) # aynı şekil, farklı
                                                            # veri türü
exact_copy = rand_nor.clone().detach() # aynı tensör

print(f"Orijinal tensör \n{rand_nor}\n")
print(f"Aynı şekil ve veri türü \n{all_sevens}\n")
print(f"Aynı şekil, farklı veri türü \n{all_ones}\n")
print(f"Tam kopya \n{exact_copy}\n")






#Tensörlere erişme (dilimleme)
import torch
const_ten = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.float32)

single_element = const_ten[1,2]
first_row = const_ten[0,:]
second_column = const_ten[:, 1]
sub_matrix = const_ten[0:2,1:3]

print(f"Tek eleman  \n{single_element}\n")
print(f"İlk sıra \n{first_row}\n")
print(f"İkinci sütun  \n{second_column}\n")
print(f"Alt matris  \n{sub_matrix}\n")


#Tensörleri yeniden şekillendirmek
import torch

all_ones = torch.zeros(2,4)

diff_shape = all_ones.reshape((2,4)) # Şekli bir liste ile belirtin

diff_shape_1 = all_ones.reshape((1,2,4))

diff_shape_2 = all_ones.reshape(-1, 2)  # Bir boyuta -1 koymak, PyTorch'a geri kalan değerlere bakarak
                                        # boyutu otomatik olarak çıkarmasını söyler


rand_t = torch.empty((2, 2, 2))
diff_shape_3 = all_ones.reshape_as(rand_t) #Başka bir tensörün şekliyle
                                           #eşleşmeyi kullanabilirsiniz


new_tensor = all_ones.clone().detach().reshape((2,4))
                                    #Başka bir tensörün şekliyle eşleşmeyi kullanabilirsiniz.                                   .
print("Şekil: (2,3)")
print(all_ones)
print("\nŞekil: (3,2)")
print(diff_shape)
print("\nŞekil: (1,2,3)")
print(diff_shape_1)
print("\nŞekil: (4,2)")
print(diff_shape_2)
print("\nŞekil: (2,2,2)")
print(diff_shape_3)
print("\nYeni tensör:")
print(new_tensor)

#Tensörleri birleştirme
import torch
all_ones = torch.ones(3,4)
all_zeros = torch.zeros_like(all_ones) # all_ones ile aynı şekil

con_hor = torch.cat([all_ones, all_zeros], dim=1) # yatay
con_ver = torch.cat([all_ones, all_zeros], dim=0) # dikey

print(f"Yatay birleştirme  \n{con_hor}\n")
print(f"Dikey birleştirme  \n{con_ver}\n")



#Matematiksel işlemler
import torch
all_ones = torch.ones(3,2, dtype=torch.float32)
all_twos = torch.full((2,3),2, dtype=torch.float32)
all_threes = torch.full((3,2),3, dtype=torch.float32)

scaler_arith = all_ones + 4
tensor_arith = all_ones - all_threes

scaler_mul = all_ones * 2
elem_mul = all_ones * all_threes
mat_mul = all_ones.matmul(all_twos)

print(f"Bir tensöre ölçekleyici ekleme \n{scaler_arith}\n")
print(f"İki tensör eklemek  \n{tensor_arith}\n")
print(f"Bir tensörün bir ölçekleyici ile çarpılması  \n{scaler_mul}\n")
print(f"Element-bilge çarpma  \n{elem_mul}\n")
print(f"Matris çarpımı  \n{mat_mul}\n")



#GPU üzerindeki tensörler
import torch

gpu_0 = torch.device('cuda') #
cpu_device = torch.device('cpu')

t1 = torch.tensor([1,2,3], device=gpu_0)
print(f"t1 on GPU 0: \n{t1}\n")
t2 = torch.tensor([1,2,3])
print(f"t2 on CPU: \n{t2}\n")
t2 = t2.to(gpu_0)
print(f"t2 on GPU 0: \n{t2}\n")
t3 = t2 + t1
t3 = t3.to(cpu_device)
print(f"t3 on GPU 0: \n{t3}\n")



#Türev Hesaplama
#Türev hesabı için gerekenler
import torch

t0 = torch.ones(3, requires_grad=True)
t1 = torch.zeros(3)

print(f"t0 \n{t0}")
print(f"t1 \n{t1}\n")

t1.requires_grad_(True)
print(f"t1 \n{t1}")
print("t1'de manuel olarak etkinleştirilen türev hesaplaması \n")
t1.requires_grad_(False)
print(f"t1 \n{t1}")
print("Manuel olarak devre dışı bıraktı ")



#Türevleri hesaplama
import torch
t1 = torch.tensor(1, dtype=torch.float32, requires_grad=True)
t2 = t1*t1 # t2, t1 cinsinden bir fonksiyondur
           # dt2/dt1 = 2*t1
t2.backward()

print(f"t1 = {t1}")
print(f"t1'e göre t2'nin türevi  = {t1.grad}")



#Daha derin işlemler
import torch

t1 = torch.tensor(1, dtype=torch.float32, requires_grad=True)

t2 = t1*t1-5 # dt2/dt1 = 2*t1

t3 = t2*2+3 # dt3/dt2 = 2

t4 = t3**4 # dt4/g3 = 4*t3^3

print(f"t1 = {t1}, t2 = {t2}, t3 = {t3}, t4 = {t4}")

t2.backward(retain_graph=True) # türevi tekrar hesaplayabilmek
                               # için "retain_graph = True" belirtiriz
print(f"t1'ye göre t2'nin gradyanı  = {t1.grad}")
# dt2/dt1 = 2 * t1
t1.grad.data.zero_() # bu türevi 0'a eşitleyecek

t3.backward(retain_graph=True)
print(f"t1'e göre t3'ün gradyanı  = {t1.grad}")
# dt3/dt1 = dt3/dt2 * dt2/dt1 = 2 * 2 * t1
t1.grad.data.zero_() # bu türevi 0'a eşitleyecek

t4.backward()
print(f"t1'e göre t4 gradyanı  = {t1.grad}")
# dt4/dt1 = dt4/dt3 * dt3/dt2 * dt2/dt1 = 4 * t3^3 * 2 * 2 * t1
t1.grad.data.zero_() # bu türevi 0'a eşitleyecek



#no_grad kullanarak tensörleri güncelleme
import torch

t1 = torch.tensor(1, dtype=torch.float32, requires_grad=True)

t2 = t1*t1-5 # dt2/dt1 = 2*t1
t2.backward() # t1'ye göre t3'ün gradyanını hesaplayın
print(f"t1 = {t1}")
print(f"t1'e göre t2'nin gradyanı  = {t1.grad.data}\n")
with torch.no_grad(): # gradyanlar hesaplanırken bu bloktaki
    t1-=t1.grad.data  # tensör işlemleri izlenmez
print(f"güncellemeden sonra t1  = {t1}")
t1.grad.data.zero_() # bu degradeyi 0'a sıfırlayacak
