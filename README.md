Zerkala

Zerkala — программа для проектирования зеркал омнидирекциональных (панорамных) камер. Мы делали её для своих роботов в RoboCup Soccer, чтобы рассчитывать зеркала под конкретные лиги и требования к обзору. Программа считает профиль зеркала по геометрическим параметрам, сразу показывает его на графике и экспортирует готовую модель для изготовления.

Возможности

Два режима расчёта профиля:
Гиперболоид — зеркало с единой точкой проекции: камера ставится в фокус гиперболы, что даёт геометрически точную развёртку без параллакса.
Эквиугловое (по Stürzl et al., ECCV 2004) — угол в сцене линейно связан с углом в камере, что даёт постоянное угловое разрешение по кадру.
Интерактивные ползунки для геометрии зеркала (для гиперболоида: угол конуса, угол пучка, полуширина, угол контрольного луча; для эквиуглового: угловое увеличение α, угол вершины γS, максимальный радиус, максимальный угол) и параметров камеры (фокусное расстояние, апертура).
Расчёт и отображение поверхностей виртуального изображения vθ/vφ и оценка размытия для эквиуглового режима.
Экспорт активного профиля в DXF (2D-профиль с аннотациями, для изготовления) и STL (тело вращения, для 3D-печати/просмотра).

Стек: Python, numpy, matplotlib, numpy-stl, ezdxf.

Zerkala

Zerkala is a design tool for omnidirectional (panoramic) camera mirrors. We built it for our robots in RoboCup Soccer, to design mirrors for specific leagues and field-of-view requirements. It computes the mirror profile from geometric parameters, plots it live, and exports a ready-to-manufacture model.

Features

Two profile modes:
Hyperboloid — a single-viewpoint mirror: the camera sits at one focus of the hyperbola, giving a geometrically exact, parallax-free unwarp.
Equiangular (after Stürzl et al., ECCV 2004) — the scene angle is linearly related to the camera angle, giving constant angular resolution across the frame.
Interactive sliders for mirror geometry (for the hyperboloid: cone angle, beam angle, half-width, control-ray angle; for the equiangular mirror: angular gain α, apex angle γS, max radius, max angle) and for camera parameters (focal length, aperture).
Computes and displays the virtual-image surfaces vθ/vφ and a blur estimate for the equiangular mode.
Exports the active profile to DXF (2D profile with annotations, for manufacturing) and STL (solid of revolution, for 3D printing/preview).

Stack: Python, numpy, matplotlib, numpy-stl, ezdxf.
