"""File to download example sky images from flickr."""
from __future__ import print_function, division
import socket
socket.setdefaulttimeout(10)
import urllib
import os
import time
import re
import traceback

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
WRITE_TO_BASE_DIR = os.path.join(FILE_DIR, "downloaded")
BASE_URL = "https://www.flickr.com/search/?text=%KEYWORD&media=photos&page=%PAGE&view_all=1"
KEYWORDS = ["sky", "night%20sky", "sunset", "sunrise"]
URLS_LIST_FILEPATH = os.path.join(WRITE_TO_BASE_DIR, "_urls.txt")
PATTERN = re.compile(r"_z\.jpg$")

def main():
    """Main function: Iterates over search pages and downloads images linked on them."""
    with open(URLS_LIST_FILEPATH, "a") as furl:
        # seems to be 25 images per page
        # by default: scan pages 1 to 250 per keyword
        for page_no in range(1, 250):
            print("<Page> %d" % (page_no))
            for keyword in KEYWORDS:
                main_url = BASE_URL.replace("%KEYWORD", keyword)
                dest_dir = os.path.join(WRITE_TO_BASE_DIR, "%s/" % (keyword,))
                try:
                    source = load_page_source(main_url, page_no)
                except Exception as exc:
                    traceback.print_exc()
                    print(exc)
                    source = None

                if source is not None:
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
                        if PATTERN.search(url):
                            print("<Image> %s" % (url))
                            try:
                                downloaded = download_image(url, dest_dir, furl)
                                if downloaded:
                                    # wait for 1sec before downloading the next image
                                    time.sleep(1.0)
                            except Exception as exc:
                                traceback.print_exc()
                                print(exc)

                # wait for 5secs before opening the next search page
                time.sleep(5.0)

def load_page_source(main_url, page_no):
    """Load the source code of a flickr search page.
    Args:
        main_url The search page url, must contain "%PAGE"
        page_no The number of the search page to request (1 ... N)
    Returns:
        Html content
    """
    url = main_url.replace("%PAGE", str(page_no))
    return " ".join(urllib.urlopen(url).readlines())

def extract_image_urls(source):
    """Finds urls to images in the source code of a flickr search page.
    Args:
        source HTML-Sourc code
    Returns:
        List of urls (strings)
    """
    # todo pylint complains here
    source = source.replace("\/", "/")
    pattern = re.compile(r"\/\/[a-zA-Z0-9]{1,4}\.staticflickr\.com\/[a-zA-Z0-9\/_]+\.(?:jpg|jpeg|png)")
    matches = re.findall(pattern, source)
    return matches

def fix_url(url):
    """Makes sure that an url is properly formatted.
    Args:
        url The url to fix
    Returns:
        The fixed URL
    """
    if url.startswith("//"):
        return "http:" + url
    else:
        return url

def download_image(source_url, dest_dir, urls_list_file):
    """Downloads an image from flickr and saves it.
    Images that were already downloaded are skipped automatically.
    
    Args:
        source_url The URL of the image.
        dest_dir The directory to save the image in.
        urls_list_file File handle for the file in which the URLs of downloaded images will be
                       saved.
    Returns:
        True if the image was downloaded
        False otherwise (including images that were skipped)
    """
    if "/" not in source_url or (".jpg" not in source_url and ".jpeg" not in source_url) or "?" in source_url:
        print("[Warning] source url '%s' is invalid" % (source_url))
        return False
    else:
        index = source_url.rfind(".com/")
        image_name = source_url[index+len(".com/"):].replace("/", "-")
        filepath = os.path.join(dest_dir, image_name)
        if os.path.isfile(filepath):
            print("[Info] skipped '%s', already downloaded" % (filepath))
            return False
        else:
            # create directory if it doesnt exist
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            # add "<URL>\t<Image-Filepath>" (without <, >) to urls file
            urls_list_file.write("%s\t%s\n" % (source_url, filepath))
            # download the image
            urllib.urlretrieve(source_url, filepath)
            return True

if __name__ == "__main__":
    main()
