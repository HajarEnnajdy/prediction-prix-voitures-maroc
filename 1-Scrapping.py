import csv
import time
import numpy as np
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
import os
import json

#hajar En-najdy
#mohamed akrame kaichy

# Configuration du WebDriver pour exécuter en mode headless
service = Service(log_path="NUL")
options = Options()
options.add_argument("--headless=new")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--log-level=3")
browser_driver = webdriver.Edge(service=service, options=options)

USER_AGENT = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
}

# Variables globales
unique_labels = set()
csv_output_file = "scraped_data.csv"
json_backup_file = "scraped_backup.json"
processed_urls = set()

def load_previous_data():
    """Charge les données sauvegardées depuis un fichier JSON de backup."""
    if os.path.exists(json_backup_file):
        with open(json_backup_file, 'r', encoding='utf-8') as f:
            backup_data = json.load(f)
            processed_urls.update(backup_data.get("processed_urls", []))
            return backup_data.get("headers", []), backup_data.get("all_records", [])
    return [], []

def save_progress(headers, data):
    """Sauvegarde l'état du scraping dans un fichier JSON de backup."""
    with open(json_backup_file, 'w', encoding='utf-8') as f:
        json.dump({
            "headers": headers,
            "all_records": data,
            "processed_urls": list(processed_urls),
            "unique_labels": list(unique_labels)
        }, f, ensure_ascii=False)

def extract_listing_links():
    """Récupère les liens des annonces non encore traitées sur la page actuelle."""
    try:
        WebDriverWait(browser_driver, 15).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "sc-1jge648-0.jZXrfL"))
        )
        elements = browser_driver.find_elements(By.CLASS_NAME, "sc-1jge648-0.jZXrfL")
        return [el.get_attribute("href") for el in elements
                if el.get_attribute("href") and el.get_attribute("href") not in processed_urls]
    except Exception as e:
        print(f"Erreur lors de l'extraction des liens: {str(e)}")
        return []

def scrape_advertisement(url):
    """Scrape les informations d'une annonce à partir de son URL."""
    try:
        response = requests.get(url, headers=USER_AGENT, timeout=10)
        if response.status_code != 200:
            return None

        page_content = BeautifulSoup(response.text, "html.parser")
        ad_data = {}

        # Extraction des détails de l'annonce
        containers = page_content.find_all('div', class_='sc-19cngu6-1')
        for container in containers:
            label = container.find('span', class_='bXFCIH')
            value = container.find('span', class_='fjZBup')
            if label and value:
                label_text = label.text.strip()
                ad_data[label_text] = value.text.strip()
                unique_labels.add(label_text)

        # Extraction du prix
        price = page_content.select_one('.sc-1x0vz2r-0.lnEFFR')
        if price:
            ad_data["Prix"] = price.text.strip().replace("DH", "").replace(" ", "")

        return ad_data

    except Exception as e:
        print(f"Erreur pour {url}: {str(e)}")
        return None

# Fonction pour réécrire le CSV
def overwrite_csv(filename, headers, data):
    """Réécrit complètement le fichier CSV avec les nouvelles données."""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)

# Fonction pour ajouter une ligne au CSV
def append_to_existing_csv(filename, headers, row):
    """Ajoute une nouvelle ligne au fichier CSV existant."""
    file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)

# Initialisation
headers, all_records = load_previous_data()
browser_driver.get("https://www.avito.ma/fr/maroc/voitures_d_occasion-%C3%A0_vendre?has_price=true")
time.sleep(3)

try:
    with open(csv_output_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Si le fichier CSV est vide, écrire les headers
        if os.path.getsize(csv_output_file) == 0 and headers:
            writer.writerow(headers)

        for page_num in range(500):  # Nombre maximal de pages à scraper
            links = extract_listing_links()
            print(f"Page {page_num+1} - {len(links)} nouveaux liens à traiter")

            for link in links:
                ad_data = scrape_advertisement(link)
                if not ad_data:
                    continue

                # Mise à jour des headers si de nouveaux champs sont trouvés
                current_headers = sorted(unique_labels | {"Prix"})
                if not headers or set(current_headers) != set(headers):
                    headers = current_headers
                    overwrite_csv(csv_output_file, headers, all_records)

                # Ajouter les données de l'annonce dans la liste
                row = [ad_data.get(h, np.nan) for h in headers]
                all_records.append(row)
                processed_urls.add(link)

                # Sauvegarde progressive
                save_progress(headers, all_records)
                append_to_existing_csv(csv_output_file, headers, row)
                print(f"✓ {link[:60]}... ajoutée avec succès.")

            # Pagination pour la page suivante
            try:
                wait = WebDriverWait(browser_driver, 10)
                next_button = wait.until(
                    EC.element_to_be_clickable((By.XPATH, "(//a[contains(@class, 'sc-2y0ggl-1') and contains(@class, 'yRCEb')])[last()]"))
                )
                next_button.click()
                print(f"✅ Passé à la page suivante ({page_num + 2})")
                time.sleep(1)
            except (NoSuchElementException, TimeoutException) as e:
                print(f"Problème pour accéder à la page suivante: {e}")
                break

            print(f"Page {page_num + 1} terminée. Nombre total d'annonces scrapées: {len(all_records)}")

except Exception as e:
    print(f"Une erreur s'est produite : {e}")

finally:
    browser_driver.quit()
    if os.path.exists(json_backup_file):
        os.remove(json_backup_file)
    print(f"Scraping terminé. Données sauvegardées dans {csv_output_file}")
