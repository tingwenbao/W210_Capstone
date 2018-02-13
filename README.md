# w210 scrapy crawlers

## Setup

```
# as root or sudoer, setup python:
apt-get update
apt-get install tmux
apt-get install python
apt-get install python-pip
# (or install Anaconda)
pip install --upgrade pip
pip install scrapy ipython
# (or install scrapy with conda)
```

## Creating new spiders

```
git clone this repo
cd to repo
scrapy startproject project_name project_name
cd project_name
scrapy list
scrapy genspider ewg ewg.org
scrapy list
```

## Example run instructions

For the ewg.org scraper:

Run with logging status messages to local file (instead of stdout):

`scrapy crawl ewg_skindeep --logfile ewg.log`

Run with saving cache to disk so that crawl can be interrupted and resumed:

`scrapy crawl ewg_skindeep -s JOBDIR=crawls/ewg_skindeep-1`

Combine the two options:

`scrapy crawl ewg_skindeep --logfile ewg.log -s JOBDIR=crawls/ewg_skindeep-1`

Running scraper(s) with `tmux` in case your SSH connection gets terminated. See <https://askubuntu.com/questions/8653/how-to-keep-processes-running-after-ending-ssh-session> for details.

For more details, see:

 * https://doc.scrapy.org/en/latest/topics/logging.html
 * https://doc.scrapy.org/en/latest/topics/jobs.html
