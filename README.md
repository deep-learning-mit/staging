# 6.s898 2023 Blogposts 

### Installation

For a hands-on walkthrough of al-folio installation, check out [this cool video tutorial](https://www.youtube.com/watch?v=g6AJ9qPPoyc) by one of the community members! 🎬 🍿

---

#### Local setup using Docker

You need to take the following steps to get `al-folio` up and running in your local machine:

- First, [install docker](https://docs.docker.com/get-docker/) (Install Docker Desktop). Make sure that Docker Destop is open/running on your computer, and if it hangs you may need to restart your computer once after the installation
- Then, clone this repository to your machine:

```bash
$ git clone git@github.com:<your-username>/<your-repo-name>.git
$ cd <your-repo-name>
```

Finally, run the following command that will pull a pre-built image from DockerHub and will run your website.

```bash
$ ./bin/dockerhub_run.sh
```

---

#### Local setup without Docker

Assuming you have [Ruby](https://www.ruby-lang.org/en/downloads/) and [Bundler](https://bundler.io/) installed on your system (*hint: for ease of managing ruby gems, consider using [rbenv](https://github.com/rbenv/rbenv)*), do the following:

```bash
$ git clone git@github.com:<your-username>/<your-repo-name>.git
$ cd <your-repo-name>
$ bundle install
$ bundle exec jekyll serve
```

---
>

## License

The theme is available as open source under the terms of the [MIT License](https://github.com/alshedivat/al-folio/blob/master/LICENSE).

Originally, **al-folio** was based on the [\*folio theme](https://github.com/bogoli/-folio) (published by [Lia Bogoev](https://liabogoev.com) and under the MIT license).
Since then, it got a full re-write of the styles and many additional cool features.

