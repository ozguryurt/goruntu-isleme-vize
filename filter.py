"""
Görüntünün hareketini algılayan ve hareket eden bölgelerde rastgele renkli partiküller oluşturan bir filtre.
"""

import cv2
import numpy as np
import random

# Video dosyası, kamera için "0" yazılacak
cap = cv2.VideoCapture('video1.mp4')

# Video veya kamera yoksa hata mesajı versin
if not cap.isOpened():
    print("Video veya kamera bulunamadı.")
    exit()

prev_frame = None  # Önceki frame
particles = []  # (x, y, lifespan) şeklinde partiküller
max_particles = 500  # Max. partikül sayısı
particle_lifespan = 15  # Partikülün görüneceği süre

while True:
    ret, frame = cap.read()
    if not ret:
        print("Görüntü alınamadı.")
        break

    # Görüntüyü gri tonlamaya çevir
    current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur ayarı
    current_frame_gray = cv2.GaussianBlur(current_frame_gray, (5, 5), 0)

    # Önceki frame'i ayarla
    if prev_frame is None:
        prev_frame = current_frame_gray
        continue

    # Videodaki hareket eden pikselleri algılamak için
    frame_diff = cv2.absdiff(current_frame_gray, prev_frame) # Şuanki kare ile önceki kare arasındaki farkı hesapla
    temp, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY) # Hareket eden pikselleri bul (25 eşik değer için)
    contours, temp = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Hareketli bölgelerin konturlarını bul

    # Hareket tespit edilen yerlere partikül ekleme kısmı
    new_particles = []  # Yeni partikülleri saklamak için liste
    for contour in contours:
        # Çok küçük konturları eklemesin
        if cv2.contourArea(contour) < 500:
            continue

        # Konturun orta noktasını bul
        M = cv2.moments(contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            # Yeni partikülü ekle (x, y, lifespan)
            new_particles.append([center_x, center_y, particle_lifespan])

    # Mevcut partiküllere yeni partikülleri teker teker ekle
    particles.extend(new_particles)

    # Partiküllerin olduğu katman
    particle_overlay = np.zeros_like(frame)

    updated_particles = []  # Güncel partikülleri saklamak için

    # Partikülleri çizme kısmı
    for particle in particles:
        x, y, lifespan = particle
        if lifespan > 0:
            lifespan -= 1  # Partikül süresi azalsın
            particle[2] = lifespan # (x, y, lifespan)
            updated_particles.append(particle)  # Partikülü güncellenmiş listeye ekle

            # Partikülün boyutu lifespan'a bağlı olarak küçülsün
            particle_size = max(1, int(lifespan / 3))

            # Partiküle rastgele renk ayarla
            particle_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            # Partikülü çiz
            cv2.circle(particle_overlay, (int(x), int(y)), particle_size, particle_color, -1)

    # Güncel partikül listesini güncelle
    particles = updated_particles

    # Max. partikül sayısı aşılmışsa eskilerini silsin
    if len(particles) > max_particles:
        particles = particles[-max_particles:]

    # Partikülleri frame'de göster
    frame = cv2.addWeighted(frame, 1, particle_overlay, 0.5, 0)

    # Görüntüyü ekranda gösterelim
    cv2.imshow("TikTok Filter", frame)

    # Önceki frame'i güncelle
    prev_frame = current_frame_gray

    # "q" tuşuna basılınca kapansın
    if cv2.waitKey(33) == ord('q'):
        break

# Kamerayı vb. kapat
cap.release()
cv2.destroyAllWindows()