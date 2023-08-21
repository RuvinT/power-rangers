#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 20:01:04 2023

@author: ruvinjagoda
"""
'''
url = https://www.ieso.ca/power-data/this-hours-data
//*[@id="demand"]/div[2]/div[1]/div/div[1]/span[2]

//*[@id="demand"]/div[2]/div[1]/div/div[2]/span
'''

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import datetime
import csv
import xml.etree.ElementTree as ET
import requests
from datetime import datetime
import os


# Set Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run Chrome in headless mode to avoid GUI requirement
    
# Set ChromeDriver executable path
chrome_driver_path = "./chromedriver"
    
# Create Chrome WebDriver instance
driver = webdriver.Chrome(executable_path=chrome_driver_path, options=chrome_options)

# URL of the webpage to scrape
url = "https://www.ieso.ca/power-data/this-hours-data"

# Open the webpage
driver.get(url)

def data_extract_scraping():
    # extract the hour
    date_hour_xpath = '//*[@id="demand"]/div[2]/div[1]/div/div[1]/span[2]'
    date_hour_element = driver.find_element(By.XPATH, date_hour_xpath)
    date_hour = date_hour_element.text
    
    # Extract the demand in MW
    demand_xpath = '//*[@id="demand"]/div[2]/div[1]/div/div[2]/span'
    demand_element = driver.find_element(By.XPATH, demand_xpath)
    demand_mw = demand_element.text
    
    # Print the extracted data
    print("Date and Hour:", date_hour)
    print("Demand in MW:", demand_mw)

    # Remove "at " and " EST" from the time string
    time_string = date_hour.replace("at ", "").replace(" EST", "").replace(".", "")
    
    # Convert the time string to a datetime object
    datetime_obj = datetime.datetime.strptime(time_string, "%I:%M %p")
    
    # Extract the hour from the datetime object
    hour = datetime_obj.hour
    
    # Adjust the hour value for midnight (12:00 AM)
    if hour == 0:
        hour = 24
    #get current date
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Demand in number
    demand_mw = demand_mw.replace(",", "").replace(" ", "").replace("MW", "")

    return current_date, hour, demand_mw


def data_extract_xml():
    # URL of the XML file
    url = "https://www.ieso.ca/-/media/files/ieso/uploaded/chart/ontario_demand_multiday.xml?la=en"
    
    # Send a GET request to the URL and retrieve the XML content
    response = requests.get(url)
    xml_content = response.text
    # Parse the XML content
    root = ET.fromstring(xml_content)
    
    # Extract the "CreatedAt" value
    created_at = root.find(".//CreatedAt").text
    # Find the "DataSet" element with Series="Actual"
    actual_dataset = root.find(".//DataSet[@Series='Actual']")
    last_value = "0"
    # Extract and print the values under the "Actual" series
    data_elements = actual_dataset.findall("./Data/Value")
    for element in data_elements:
        value = element.text
        if(not(value is None)):
            last_value = value 
            
        
        
    # Print the "CreatedAt" value
    print("Created At:", created_at)
    # Convert the datetime string to a datetime object
    datetime_obj = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S")
    
    # Extract the date and hour
    date = datetime_obj.date()
    hour = datetime_obj.hour
    
    # Print the extracted date and hour
    print("Date:", date)
    print("Hour:", hour)
    print("Value:", last_value)
    
    return date,hour,last_value

def append_new_data_to_csv(current_date,hour, demand_mw):
    # Create a list of data for the CSV file
    data = [current_date, str(hour), demand_mw]
    
    # Define the CSV file path
    csv_file_path = "PUB_Demand.csv"
    
    # Check if the CSV file already exists
    csv_file_exists = os.path.isfile(csv_file_path)
    
    # Read the existing data from the CSV file
    existing_data = []
    if csv_file_exists:
        with open(csv_file_path, mode='r') as file:
            reader = csv.reader(file)
            existing_data = list(reader)
    
    # Append the data after the last row
    existing_data.append(data)
    
    # Write the data to the CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
    
        # Write the existing data
        writer.writerows(existing_data)
        
        
def hourly_run_task():
    
    current_date,hour, demand_mw = None
    
    try:
        
        current_date,hour, demand_mw = data_extract_scraping() 
                
        raise Exception("Exception occurred in scraping")
    except Exception as e:
        print("Exception occurred:", e)
        
        current_date,hour, demand_mw = data_extract_xml() 
    # Close the browser
    driver.quit()    
    append_new_data_to_csv(current_date,hour, demand_mw)