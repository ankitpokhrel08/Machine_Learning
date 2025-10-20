from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime, timedelta
import time
import os

# Initialize driver
s = Service('/opt/homebrew/bin/chromedriver')
driver = webdriver.Chrome(service=s)
wait = WebDriverWait(driver, 20)  # Increase wait time to 20 seconds
driver.maximize_window()  # Maximize window to ensure all elements are visible

# Open target website
driver.get("https://kalimatimarket.gov.np/price")
time.sleep(2)

# Output file
output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kalimatidata.txt")

# Create the file first (clear it if it exists)
with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"Kalimati Market Data: June 1-15, 2025\n\n")

# Date range: June 1 to June 15, 2025
start_date = datetime(2025, 6, 1)
end_date = datetime(2025, 6, 15)

date = start_date
while date <= end_date:
    # Change date format to DD/MM/YYYY which is more commonly accepted by date pickers
    date_str = date.strftime("%d/%m/%Y")

    try:
        # Input date
        date_input = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="datePricing"]')))
        driver.execute_script("arguments[0].value = '';", date_input)
        date_input.send_keys(date_str)
        time.sleep(1)  # Add short delay after entering date

        # Click the button
        search_button = driver.find_element(By.XPATH, '//*[@id="queryFormDues"]/div/div[2]/button')
        search_button.click()
        
        # Wait longer for table to load with explicit wait
        print(f"Waiting for data to load for {date_str}...")
        time.sleep(3)  # Give more time for AJAX to complete
        
        try:
            # First try with ID
            table = wait.until(EC.visibility_of_element_located((By.ID, "priceDailyTable")))
            table_html = table.get_attribute("outerHTML")
        except:
            # If ID fails, try alternate selectors
            print("Table ID not found, trying alternate method...")
            # Try to get the table by class or tag
            tables = driver.find_elements(By.TAG_NAME, "table")
            if tables:
                table_html = tables[0].get_attribute("outerHTML")
            else:
                # If no tables found, get the whole content area
                content_area = driver.find_element(By.CLASS_NAME, "content-area")
                table_html = content_area.get_attribute("outerHTML")

        # Save to file
        html = driver.page_source
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"--- Data for {date.strftime('%Y-%m-%d')} ---\n")
            f.write(html)

        print(f"[✓] Scraped data for {date.strftime('%Y-%m-%d')}")

    except Exception as e:
        print(f"[!] Failed for {date.strftime('%Y-%m-%d')}: {str(e)}")
        # Try to take a screenshot on error for debugging
        try:
            screenshot_path = f"error_{date.strftime('%Y%m%d')}.png"
            driver.save_screenshot(screenshot_path)
            print(f"Screenshot saved to {screenshot_path}")
        except:
            pass

    date += timedelta(days=1)
    time.sleep(2)  # Add delay between iterations

driver.quit()
