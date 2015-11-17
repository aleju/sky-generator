import socket
socket.setdefaulttimeout(5)
import urllib
import os
import time
import re
import traceback

MAIN_URLS = [
    "https://www.flickr.com/search/?text=sky&media=photos&page=%PAGE&view_all=1",
    "https://www.flickr.com/search/?text=night%20sky&media=photos&page=%PAGE&view_all=1",
    "https://www.flickr.com/search/?text=sunset&media=photos&page=%PAGE&view_all=1",
    "https://www.flickr.com/search/?text=sunrise&media=photos&page=%PAGE&view_all=1"
]
DEST_DIR = "orig"
URLS_LIST_FILEPATH = "_urls.txt"
PATTERN = re.compile(r"_z\.jpg$")

def main():
    with open(URLS_LIST_FILEPATH, "a") as fhandle:
        # seems to be 25 images per page
        for page_no in range(1, 10000):
            print("<Page> %d" % (page_no))
            for main_url in MAIN_URLS:
                source = load_page_source(main_url, page_no)
                raw_urls = extract_image_urls(source)
                urls = set([fix_url(url) for url in raw_urls])
                for url in urls:
                    # Examples:
                    # http://c3.staticflickr.com/3/2935/14030724164_8a4cf0b48d.jpg   (500x333, 1.5:1)
                    # http://c3.staticflickr.com/3/2935/14030724164_8a4cf0b48d_n.jpg (320x213, 1.5:1, 64%)
                    # http://c3.staticflickr.com/3/2935/14030724164_8a4cf0b48d_q.jpg (150x150, 1.0:1, 30%)
                    # http://c3.staticflickr.com/3/2935/14030724164_8a4cf0b48d_m.jpg (240x160, 1.5:1, 48%)
                    # http://c3.staticflickr.com/3/2935/14030724164_8a4cf0b48d_s.jpg ( 75x75,  1.0:1, 15%)
                    # http://c3.staticflickr.com/3/2935/14030724164_8a4cf0b48d_t.jpg (100x67,  1.5:1, 20%)
                    # http://c3.staticflickr.com/3/2935/14030724164_8a4cf0b48d_z.jpg (640x427, 1.5:1, 128%)
                    #if "_n" in url or "_t" in url:
                    if PATTERN.search(url):
                        print("<Image> %s" % (url))
                        try:
                            download_image(url, DEST_DIR, fhandle)
                        except Exception as e:
                            traceback.print_exc()
                            print(e)
                        time.sleep(1.0)
                time.sleep(5.0)

def load_page_source(main_url, page_no):
    url = main_url.replace("%PAGE", str(page_no))
    return " ".join(urllib.urlopen(url).readlines())

def extract_image_urls(source):
    source = source.replace("\/", "/")
    pattern = re.compile(r"\/\/[a-zA-Z0-9]{1,4}\.staticflickr\.com\/[a-zA-Z0-9\/_]+\.(?:jpg|jpeg|png)")
    matches = re.findall(pattern, source)
    return matches

def fix_url(url):
    if url.startswith("//"):
        return "http:" + url
    else:
        return url

def download_image(source_url, dest_dir, urls_list_file):
    if "/" not in source_url or (".jpg" not in source_url and ".jpeg" not in source_url) or "?" in source_url:
        print("[Warning] source url '%s' is invalid" % (source_url))
    else:
        index = source_url.rfind(".com/")
        image_name = source_url[index+len(".com/"):].replace("/", "-")
        filepath = os.path.join(dest_dir, image_name)
        if os.path.isfile(filepath):
            print("[Info] skipped '%s', already downloaded" % (filepath))
        else:
            urls_list_file.write("%s\t%s\n" % (source_url, image_name))
            urllib.urlretrieve(source_url, filepath)

if __name__ == "__main__":
    main()
