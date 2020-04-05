<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!--
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
-->

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/LisaWillig/udkm_CamViewer">
    <img src="GUI/Icons/beamstab.png" alt="Logo" width="80" height="80">
  </a>
  <h3 align="center">udkm_CamViewer</h3>
  <p align="center">
    Application used for viewing and analysing live camera feeds in a laboratory environment. 

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Roadmap](#roadmap)
* [License](#license)
* [Contact](#contact)

<!-- ABOUT THE PROJECT -->
## About The Project



The application was designed for reading, viewing and analysing camera images in a laboratory environment. That includes the usage as a webcam to observe the sample under experimental conditions, or as a beam profiler to analyse the properties of a laser spot. In this context a fluence calculater is also implemented (energy distributed from a laser beam onto a sample surface depending on time and area).

The settings for each camera are saved and loaded when it is next used.

It was mainly developed with cameras from the company Basler, but an interface to Thorlabs CCDs does exist as well.

![Application Cam Screenshot][cam-screenshot]

![Application Analysis Screenshot][analysis-screenshot]

### Built With

* Python 
* PyQt 5 as UI framework
* pyqtgraph as live data display


<!-- GETTING STARTED 
## Getting Started
To get a local copy up and running follow these simple steps.
### Prerequisites
This is an example of how to list things you need to use the software and how to install them.
* npm
```sh
npm install npm@latest -g
```
### Installation
1. Clone the repo
```sh
git clone https://github.com/LisaWillig/udkm_CamViewer.git
```
2. Install NPM packages
```sh
npm install
```
<!-- USAGE EXAMPLES 
## Usage
Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.
_For more examples, please refer to the [Documentation](https://example.com)_
-->

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/LisaWillig/udkm_CamViewer/issues) for a list of proposed features (and known issues).


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact

Lisa Willig - lisa.willig.de@gmail.com

Project Link: [https://github.com/LisaWillig/udkm_CamViewer](https://github.com/LisaWillig/udkm_CamViewer)



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links 
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=flat-square
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=flat-square
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=flat-square
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=flat-square
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt-->
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/LisaWillig
[cam-screenshot]: images/CamUse.png
[analysis-screenshot]: images/ProfilerUse.png
