# guimodule.py

import customtkinter as ctk
from tkinter import filedialog, messagebox
from processingmanager import ProcessingManager
from configmodule import Config

config = Config()

class GUIManager(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Pencere ayarları
        self.title("PDF İşleme Aracı")
        self.geometry("800x600")
        self.processing_manager = ProcessingManager()

        # Tema ayarları
        theme = config.GUITHEME or "light"
        ctk.set_appearance_mode(theme)
        ctk.set_default_color_theme("blue")

        # Ana çerçeve
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Başlık
        self.title_label = ctk.CTkLabel(self.main_frame, text="PDF İşleme Aracı", font=("Arial", 24))
        self.title_label.pack(pady=10)

        # PDF seçme düğmesi
        self.select_pdf_button = ctk.CTkButton(
            self.main_frame, text="PDF Seç", command=self.select_pdf_file
        )
        self.select_pdf_button.pack(pady=10)

        # İşlem başlatma düğmesi
        self.start_process_button = ctk.CTkButton(
            self.main_frame, text="İşlemi Başlat", command=self.start_processing, state="disabled"
        )
        self.start_process_button.pack(pady=10)

        # Durum etiketi
        self.status_label = ctk.CTkLabel(self.main_frame, text="Durum: Bekleniyor...", font=("Arial", 14))
        self.status_label.pack(pady=20)

    def select_pdf_file(self):
        """
        PDF dosyasını seçmek için bir dosya seçici açar.
        """
        file_path = filedialog.askopenfilename(
            title="PDF Dosyası Seç",
            filetypes=[("PDF Files", "*.pdf")]
        )
        
        if file_path:
            self.pdf_path = file_path
            self.status_label.configure(text=f"Seçilen Dosya: {file_path}")
            self.start_process_button.configure(state="normal")
            config.logger.info(f"PDF dosyası seçildi: {file_path}")
        else:
            messagebox.showwarning("Uyarı", "Bir PDF dosyası seçmediniz.")

    def start_processing(self):
        """
        Seçilen PDF dosyasını işler ve sonuçları kaydeder.
        """
        try:
            if not hasattr(self, "pdf_path"):
                messagebox.showerror("Hata", "Lütfen önce bir PDF dosyası seçin.")
                return

            self.status_label.configure(text="İşlem başlatılıyor...")
            config.logger.info(f"İşlem başlatıldı: {self.pdf_path}")

            # PDF işleme işlemi
            results = self.processing_manager.process_pdf(self.pdf_path)

            if results:
                pdf_id = self.pdf_path.split("/")[-1].split(".")[0]
                self.processing_manager.save_data_to_chromadb(results, pdf_id)
                messagebox.showinfo("Başarılı", "PDF başarıyla işlendi ve veriler kaydedildi.")
                config.logger.info(f"İşlem tamamlandı: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem tamamlandı.")
            else:
                messagebox.showerror("Hata", "PDF işlenirken bir hata oluştu.")
                config.logger.error(f"PDF işlenemedi: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem başarısız.")
        
        except Exception as e:
            messagebox.showerror("Hata", f"Bir hata oluştu: {e}")
            config.logger.error(f"İşlem sırasında hata oluştu: {e}")

# GUI'yi çalıştırmak için:
if __name__ == "__main__":
    app = GUIManager()
    app.mainloop()
