# social-media-analysis

Bu projede artan kullanıcı yorumlarına çözüm olarak yorumları gruplayacak, sınıflandıracak ve konu üzerine görüşleri özetleyecek bir sistem kurmak amaçlanmıştır.
Öncelikle veriler analiz edilip temizlenmiş Topic'ler iççin embedding işlemi ile Qdrant vektör veriitabanına yüklenmiştir. Qdrant ve Postgresql kurulumları docker ile  eklenmiştir.
Sınıflandırma işlemi için Distilbert modeli eğitilmiş ve model  kaydedilmiştir. Özet üretimi için Bart modeli kullanılmıştır.
Modellerin kullanımları ve analizleri notebooks/ klasöründe yapılmış çıktılar data/ ve doc/ klasöründe saklanmıştır.
event-driven mimari için GRPC ve RabbitMq sistemleri kurulacak olup version-2'de eklenebilecektir.
