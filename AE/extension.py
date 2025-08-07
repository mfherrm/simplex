import cadenzaanalytics as ca
import os
from AE.url_response import UrlResponse

# Umgebungsvariable:

WEBSERVICE_HOST= os.getenv('VISUALISATION_HOST')



urlpart = f"{WEBSERVICE_HOST}/"

# -----------------------------------------------------------------------------------------------------
# 2.5D Gebäude aus DB in Cadenza individuell Färbung mit WMS-Hintergrund
# -----------------------------------------------------------------------------------------------------
def specific_analytics_function(meta_data, data):
    #print(meta_data)
    
    
    rechenlauf= data.iloc[0, 0].astype(str)
    
    maxData=0
    # Zuerst konvertieren wir die letzte Spalte in Strings

    if data.shape[1] == 2:
        # Konvertiere die letzte Spalte in einen String-Datentyp
        letzte_spalte_als_string = data.iloc[:, -1].astype(str)

        # Zähle die Anzahl der Elemente in der letzten Spalte
        anzahl_elemente = len(letzte_spalte_als_string)

        # Prüfe, ob mehr als 130 Elemente vorhanden sind
        if anzahl_elemente > 130:
            maxData=1
            # Verwende nur die ersten 130 Elemente, falls mehr als 130 vorhanden sind
            letzte_spalte_als_string = letzte_spalte_als_string.iloc[:130]
            
        gebaeude_komma_getrennte_string = ','.join(letzte_spalte_als_string)
    else:
        gebaeude_komma_getrennte_string ='0'
    
    url = f"{urlpart}app/2d-db-geb-eigfaerb-wms?rechenlauf={rechenlauf}&gebid={gebaeude_komma_getrennte_string}&max={maxData}"
    return UrlResponse(url)
    
    



image_attribute_group = ca.AttributeGroup(
    name="Simulationslauf",
    print_name="Simulationslauf ID",
    data_types=[ca.DataType.INT64],
    min_attributes= 1,
    max_attributes= 1
)

image_attribute_group1 = ca.AttributeGroup(
    name="Gebäude",
    print_name="Gebäude ID",
    data_types=[ca.DataType.INT64], 
    min_attributes= 0,
    max_attributes= 1
)

image_extension_individeulleGeb = ca.CadenzaAnalyticsExtension(
    relative_path="2d-individuelle-gebaeudefaerbung-wms",
    analytics_function=specific_analytics_function,
    print_name="2.5D individuelle Gebaeudefärbung mit WMS",
    extension_type=ca.ExtensionType.VISUALIZATION,
    attribute_groups=[image_attribute_group, image_attribute_group1]
)

# -----------------------------------------------------------------------------------------------------
# 2.5D alle Gebäud aus DB in Cadenza eingeferbt mit WMS-Hintergrund
# -----------------------------------------------------------------------------------------------------
def specific_analytics_function_alleGeb(meta_data, data):
    
    rechenlauf= data.iloc[0, 0].astype(str)
    
    url = f"{urlpart}app/2d-db-geb-allefaerbung-wms?rechenlauf={rechenlauf}"
    return UrlResponse(url)



image_extension_alleGeb = ca.CadenzaAnalyticsExtension(
    relative_path="2d-alle-gebaeude-eingefaerbt-wms",
    analytics_function=specific_analytics_function_alleGeb,
    print_name="2.5D alle Gebäude eingefärbt mit WMS",
    extension_type=ca.ExtensionType.VISUALIZATION,
    attribute_groups=[image_attribute_group]
)

# -----------------------------------------------------------------------------------------------------
# 2.5D alle Gebäude aus DB in Cadenza eingeferbt mit WMS-Hintergrund mit cookie
# -----------------------------------------------------------------------------------------------------
def specific_analytics_function_alleGeb_cookie(meta_data, data):
    
    rechenlauf= data.iloc[0, 0].astype(str)
    
    url = f"{urlpart}app/2d-db-geb-allefaerbung-wms-cookie?rechenlauf={rechenlauf}"
    return UrlResponse(url)



image_extension_alleGeb_cookie = ca.CadenzaAnalyticsExtension(
    relative_path="2d-alle-gebaeude-eingefaerbt-wms-cookie",
    analytics_function=specific_analytics_function_alleGeb_cookie,
    print_name="2.5D alle Gebäude eingefärbt mit WMS und Szenenerhalt",
    extension_type=ca.ExtensionType.VISUALIZATION,
    attribute_groups=[image_attribute_group]
)


# -----------------------------------------------------------------------------------------------------
# 2.5D alle basemap Gebäude in Cadenza individuell färben
# -----------------------------------------------------------------------------------------------------
def specific_analytics_function_basmap2D(meta_data, data):
    #print(meta_data)
    
  
    rechenlauf= data.iloc[0, 0].astype(str)
    maxData=0
    # Zuerst konvertieren wir die letzte Spalte in Strings
   
    if data.shape[1] == 2:
        # Konvertiere die letzte Spalte in einen String-Datentyp
        letzte_spalte_als_string = data.iloc[:, -1].astype(str)

        # Zähle die Anzahl der Elemente in der letzten Spalte
        anzahl_elemente = len(letzte_spalte_als_string)

        # Prüfe, ob mehr als 130 Elemente vorhanden sind
        if anzahl_elemente > 130:
            maxData=1
            # Verwende nur die ersten 130 Elemente, falls mehr als 130 vorhanden sind
            letzte_spalte_als_string = letzte_spalte_als_string.iloc[:130]
            
        gebaeude_komma_getrennte_string = ','.join(letzte_spalte_als_string)
    else:
        gebaeude_komma_getrennte_string ='0'
    
     
    url= f"{urlpart}app/2d-basemap-geb?rechenlauf={rechenlauf}&gebid={gebaeude_komma_getrennte_string}&max={maxData}"
    return UrlResponse(url)
    

image_extension_individeulleGebbasmap = ca.CadenzaAnalyticsExtension(
    relative_path="25d-gebaeude-basmape-individuelle-faerbung",
    analytics_function=specific_analytics_function_basmap2D,
    print_name="2.5D Gebäudem basmape individuelle Färbung",
    extension_type=ca.ExtensionType.VISUALIZATION,
    attribute_groups=[image_attribute_group, image_attribute_group1]
)

# -----------------------------------------------------------------------------------------------------
# 3D alle basemap Gelände und basemap Gebäude  
# -----------------------------------------------------------------------------------------------------
def specific_analytics_function_basmap3D(meta_data, data):
    #print(meta_data)
    
   
    rechenlauf= data.iloc[0, 0].astype(str)
    maxData=0
    # Zuerst konvertieren wir die letzte Spalte in Strings
    
    if data.shape[1] == 2:
        # Konvertiere die letzte Spalte in einen String-Datentyp
        letzte_spalte_als_string = data.iloc[:, -1].astype(str)

        # Zähle die Anzahl der Elemente in der letzten Spalte
        anzahl_elemente = len(letzte_spalte_als_string)

        # Prüfe, ob mehr als 130 Elemente vorhanden sind
        if anzahl_elemente > 130:
            maxData=1
            # Verwende nur die ersten 130 Elemente, falls mehr als 130 vorhanden sind
            letzte_spalte_als_string = letzte_spalte_als_string.iloc[:130]
            
        gebaeude_komma_getrennte_string = ','.join(letzte_spalte_als_string)
    else:
        gebaeude_komma_getrennte_string ='0'
    
     
    url= f"{urlpart}app/3d-base-gel-base-geb?rechenlauf={rechenlauf}&gebid={gebaeude_komma_getrennte_string}&max={maxData}"
    return UrlResponse(url)


image_extension_3D_Gelaende_basamp_Gebaeude_basmap = ca.CadenzaAnalyticsExtension(
    relative_path="3d-gelaende-basamp-gebaeude-basmap",
    analytics_function=specific_analytics_function_basmap3D,
    print_name="3D Gelände basamp Gebäude basmap",
    extension_type=ca.ExtensionType.VISUALIZATION,
    attribute_groups=[image_attribute_group, image_attribute_group1]
)

# -----------------------------------------------------------------------------------------------------
# 3D alle basemap Geblände und DB Gebäude und ividuelle färbung
# -----------------------------------------------------------------------------------------------------
def specific_analytics_function_gebindividuell3D(meta_data, data):
    #print(meta_data)
    
   
    rechenlauf= data.iloc[0, 0].astype(str)
    maxData=0
    # Zuerst konvertieren wir die letzte Spalte in Strings
   
    if data.shape[1] == 2:
        # Konvertiere die letzte Spalte in einen String-Datentyp
        letzte_spalte_als_string = data.iloc[:, -1].astype(str)

        # Zähle die Anzahl der Elemente in der letzten Spalte
        anzahl_elemente = len(letzte_spalte_als_string)

        # Prüfe, ob mehr als 130 Elemente vorhanden sind
        if anzahl_elemente > 130:
            maxData=1
            # Verwende nur die ersten 130 Elemente, falls mehr als 130 vorhanden sind
            letzte_spalte_als_string = letzte_spalte_als_string.iloc[:130]
            
        gebaeude_komma_getrennte_string = ','.join(letzte_spalte_als_string)
    else:
        gebaeude_komma_getrennte_string ='0'
    
     
    url= f"{urlpart}app/3d-basemap-gel-db-geb?rechenlauf={rechenlauf}&gebid={gebaeude_komma_getrennte_string}&max={maxData}"
    return UrlResponse(url)


image_extension_3D_Gelaende_basamp_Gebaeude_DB = ca.CadenzaAnalyticsExtension(
    relative_path="3d-gelaende-basamp-gebaeude-db",
    analytics_function=specific_analytics_function_gebindividuell3D,
    print_name="3D Gelände basamp Gebäude DB",
    extension_type=ca.ExtensionType.VISUALIZATION,
    attribute_groups=[image_attribute_group, image_attribute_group1]
)

# -----------------------------------------------------------------------------------------------------
# 3D individuelle Geblände und DB Gebäude  
# -----------------------------------------------------------------------------------------------------
def specific_analytics_function_allindividuell3D(meta_data, data):
    #print(meta_data)
    
    
    rechenlauf= data.iloc[0, 0].astype(str)
    maxData=0
    # Zuerst konvertieren wir die letzte Spalte in Strings
   
    if data.shape[1] == 2:
        # Konvertiere die letzte Spalte in einen String-Datentyp
        letzte_spalte_als_string = data.iloc[:, -1].astype(str)

        # Zähle die Anzahl der Elemente in der letzten Spalte
        anzahl_elemente = len(letzte_spalte_als_string)

        # Prüfe, ob mehr als 130 Elemente vorhanden sind
        if anzahl_elemente > 130:
            maxData=1
            # Verwende nur die ersten 130 Elemente, falls mehr als 130 vorhanden sind
            letzte_spalte_als_string = letzte_spalte_als_string.iloc[:130]
            
        gebaeude_komma_getrennte_string = ','.join(letzte_spalte_als_string)
    else:
        gebaeude_komma_getrennte_string ='0'
    
     
    url= f"{urlpart}app/3d-eigen-gelaende-db-geb?rechenlauf={rechenlauf}&gebid={gebaeude_komma_getrennte_string}&max={maxData}"
    return UrlResponse(url)


image_extension_3D_Gelaende_eigen_Gebaeude_DB = ca.CadenzaAnalyticsExtension(
    relative_path="3d-gelaende-eigen-gebaeude-db",
    analytics_function=specific_analytics_function_allindividuell3D,
    print_name="3D Gelände eigen Gebäude DB",
    extension_type=ca.ExtensionType.VISUALIZATION,
    attribute_groups=[image_attribute_group, image_attribute_group1]
)

# -----------------------------------------------------------------------------------------------------
# 3D WMS Simualtionsdaten und basmap Geblände und individuelle Gebäude  
# -----------------------------------------------------------------------------------------------------
def specific_analytics_function_WMS3D(metadata: ca.RequestMetadata, data):
    #print(meta_data)
    
    
    rechenlauf= data.iloc[0, 0].astype(str)
    print (rechenlauf)
    maxData=0
    # Zuerst konvertieren wir die letzte Spalte in Strings
   
    if data.shape[1] == 2:
        # Konvertiere die letzte Spalte in einen String-Datentyp
        letzte_spalte_als_string = data.iloc[:, -1].astype(str)

        # Zähle die Anzahl der Elemente in der letzten Spalte
        anzahl_elemente = len(letzte_spalte_als_string)

        # Prüfe, ob mehr als 130 Elemente vorhanden sind
        if anzahl_elemente > 130:
            maxData=1
            # Verwende nur die ersten 130 Elemente, falls mehr als 130 vorhanden sind
            letzte_spalte_als_string = letzte_spalte_als_string.iloc[:130]
            
        gebaeude_komma_getrennte_string = ','.join(letzte_spalte_als_string)
    else:
        gebaeude_komma_getrennte_string ='0'
    
    url= f"{urlpart}app/3d-wms-base-gel-db-geb?rechenlauf={rechenlauf}&gebid={gebaeude_komma_getrennte_string}&max={maxData}"
    return UrlResponse(url)
    

image_extension_3D_WMS = ca.CadenzaAnalyticsExtension(
    relative_path="3d-wms-simulationsdaten",
    analytics_function=specific_analytics_function_WMS3D,
    print_name="3D WMS Simulationsdaten",
    extension_type=ca.ExtensionType.VISUALIZATION,
    attribute_groups=[image_attribute_group, image_attribute_group1]
)


# -----------------------------------------------------------------------------------------------------
# 3D WMS Simualtionsdaten und basmap Geblände und individuelle Gebäude mit Cookies 
# -----------------------------------------------------------------------------------------------------
def specific_analytics_function_WMS3D_cookie(metadata: ca.RequestMetadata, data):
    #print(meta_data)
    
    
    rechenlauf= data.iloc[0, 0].astype(str)
    print (rechenlauf)
    maxData=0
    # Zuerst konvertieren wir die letzte Spalte in Strings
   
    if data.shape[1] == 2:
        # Konvertiere die letzte Spalte in einen String-Datentyp
        letzte_spalte_als_string = data.iloc[:, -1].astype(str)

        # Zähle die Anzahl der Elemente in der letzten Spalte
        anzahl_elemente = len(letzte_spalte_als_string)

        # Prüfe, ob mehr als 130 Elemente vorhanden sind
        if anzahl_elemente > 130:
            maxData=1
            # Verwende nur die ersten 130 Elemente, falls mehr als 130 vorhanden sind
            letzte_spalte_als_string = letzte_spalte_als_string.iloc[:130]
            
        gebaeude_komma_getrennte_string = ','.join(letzte_spalte_als_string)
    else:
        gebaeude_komma_getrennte_string ='0'
    
    url= f"{urlpart}app/3d-wms-base-gel-db-geb-co?rechenlauf={rechenlauf}&gebid={gebaeude_komma_getrennte_string}&max={maxData}"
    return UrlResponse(url)
    

image_extension_3D_WMS_cookie = ca.CadenzaAnalyticsExtension(
    relative_path="3d-wms-simulationsdaten-cookie",
    analytics_function=specific_analytics_function_WMS3D_cookie,
    print_name="3D WMS Simulationsdaten mit Szenenerhalt",
    extension_type=ca.ExtensionType.VISUALIZATION,
    attribute_groups=[image_attribute_group, image_attribute_group1]
)
# -----------------------------------------------------------------------------------------------------
# 3D WMS Simualtionsdaten und basmap Geblände und individuelle Gebäude mit Cookies alle Gebäude eingefärbt
# -----------------------------------------------------------------------------------------------------
def specific_analytics_function_WMS3D_cookie_alleGeb(meta_data, data):
    
    rechenlauf= data.iloc[0, 0].astype(str)
    
    url = f"{urlpart}app/3d-wms-base-gel-db-geb-alleFaerbung-co?rechenlauf={rechenlauf}"
    return UrlResponse(url)



image_extension_3D_WMS_alleGeb_cookie = ca.CadenzaAnalyticsExtension(
    relative_path="3d-wms-simulationsdaten-allefaerbung-cookie",
    analytics_function=specific_analytics_function_WMS3D_cookie_alleGeb,
    print_name="3D WMS Simulationsdaten alle Gebäude eingefärbt mit Szenenerhalt",
    extension_type=ca.ExtensionType.VISUALIZATION,
    attribute_groups=[image_attribute_group]
)

# -----------------------------------------------------------------------------------------------------
# 3D WMS Simualtionsdaten und basmap Geblände und individuelle Gebäude alle Gebäude eingefärbt
# -----------------------------------------------------------------------------------------------------
def specific_analytics_function_WMS3D_alleGeb(meta_data, data):
    
    rechenlauf= data.iloc[0, 0].astype(str)
    
    url = f"{urlpart}app/3d-wms-base-gel-db-geb-alleFaerbung?rechenlauf={rechenlauf}"
    return UrlResponse(url)



image_extension_3D_WMS_alleGeb = ca.CadenzaAnalyticsExtension(
    relative_path="3d-wms-simulationsdaten-allefaerbung",
    analytics_function=specific_analytics_function_WMS3D_alleGeb,
    print_name="3D WMS Simulationsdaten alle Gebäude eingefärbt",
    extension_type=ca.ExtensionType.VISUALIZATION,
    attribute_groups=[image_attribute_group]
)

# -----------------------------------------------------------------------------------------------------
# 2.5D polygone Simualtionsdaten und individuelle Gebäudeeinfärbung ohne WMS-Hintergrund
# -----------------------------------------------------------------------------------------------------
def specific_analytics_function_ohenWMS(metadata: ca.RequestMetadata, data):
    #print(meta_data)
    
    
    rechenlauf= data.iloc[0, 0].astype(str)
    print (rechenlauf)
    maxData=0
    # Zuerst konvertieren wir die letzte Spalte in Strings
   
    if data.shape[1] == 2:
        # Konvertiere die letzte Spalte in einen String-Datentyp
        letzte_spalte_als_string = data.iloc[:, -1].astype(str)

        # Zähle die Anzahl der Elemente in der letzten Spalte
        anzahl_elemente = len(letzte_spalte_als_string)

        # Prüfe, ob mehr als 130 Elemente vorhanden sind
        if anzahl_elemente > 130:
            maxData=1
            # Verwende nur die ersten 130 Elemente, falls mehr als 130 vorhanden sind
            letzte_spalte_als_string = letzte_spalte_als_string.iloc[:130]
            
        gebaeude_komma_getrennte_string = ','.join(letzte_spalte_als_string)
    else:
        gebaeude_komma_getrennte_string ='0'
    
    url= f"{urlpart}app/2d-db-geb-eigfaerb?rechenlauf={rechenlauf}&gebid={gebaeude_komma_getrennte_string}&max={maxData}"
    return UrlResponse(url)

image_extension_individuelleGeb_ohenWMS = ca.CadenzaAnalyticsExtension(
    relative_path="25d-individuelle-gebaeudefaerbung-ohne-wms",
    analytics_function=specific_analytics_function_ohenWMS,
    print_name="2.5D individuelle Gebaeudefärbung ohne WMS",
    extension_type=ca.ExtensionType.VISUALIZATION,
    attribute_groups=[image_attribute_group, image_attribute_group1]
)

# -----------------------------------------------------------------------------------------------------
# Punktwoke
# -----------------------------------------------------------------------------------------------------
def specific_analytics_Punktwolke(meta_data, data):
    url = f"{urlpart}app/punktwolke"
    return UrlResponse(url)



image_extension_punktwolke = ca.CadenzaAnalyticsExtension(
    relative_path="punktwolke",
    analytics_function=specific_analytics_Punktwolke,
    print_name="Punktwolke",
    extension_type=ca.ExtensionType.VISUALIZATION,
    attribute_groups=[]
)


analytics_service = ca.CadenzaAnalyticsExtensionService()
analytics_service.add_analytics_extension(image_extension_individeulleGeb)
analytics_service.add_analytics_extension(image_extension_alleGeb)
analytics_service.add_analytics_extension(image_extension_individeulleGebbasmap)
analytics_service.add_analytics_extension(image_extension_3D_Gelaende_basamp_Gebaeude_basmap)
analytics_service.add_analytics_extension(image_extension_3D_Gelaende_basamp_Gebaeude_DB)
analytics_service.add_analytics_extension(image_extension_3D_Gelaende_eigen_Gebaeude_DB)
analytics_service.add_analytics_extension(image_extension_3D_WMS)
analytics_service.add_analytics_extension(image_extension_3D_WMS_alleGeb)
analytics_service.add_analytics_extension(image_extension_individuelleGeb_ohenWMS)
analytics_service.add_analytics_extension(image_extension_alleGeb_cookie)
analytics_service.add_analytics_extension(image_extension_3D_WMS_cookie)
analytics_service.add_analytics_extension(image_extension_3D_WMS_alleGeb_cookie)
analytics_service.add_analytics_extension(image_extension_punktwolke)
# run development server, remove in production environment
if __name__ == '__main__':
    analytics_service.run_development_server(5005)
