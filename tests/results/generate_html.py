#!/usr/bin/env python3
"""
Convert JUnit XML and Coverage XML to HTML reports
No external dependencies - uses only Python stdlib
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime


def parse_junit_xml(xml_path):
    """Parse JUnit XML and extract test results"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    results = {
        'total': 0,
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'errors': 0,
        'time': 0.0,
        'testcases': []
    }

    # Handle both <testsuites> and single <testsuite>
    testsuites = root.findall('.//testsuite')
    if not testsuites:
        testsuites = [root] if root.tag == 'testsuite' else []

    for testsuite in testsuites:
        for testcase in testsuite.findall('testcase'):
            classname = testcase.get('classname', '')
            name = testcase.get('name', '')
            time = float(testcase.get('time', 0))

            failure = testcase.find('failure')
            error = testcase.find('error')
            skipped = testcase.find('skipped')

            status = 'passed'
            message = ''

            if failure is not None:
                status = 'failed'
                message = failure.get('message', '')
                results['failed'] += 1
            elif error is not None:
                status = 'error'
                message = error.get('message', '')
                results['errors'] += 1
            elif skipped is not None:
                status = 'skipped'
                message = skipped.get('message', '')
                results['skipped'] += 1
            else:
                results['passed'] += 1

            results['total'] += 1
            results['time'] += time

            results['testcases'].append({
                'classname': classname,
                'name': name,
                'time': time,
                'status': status,
                'message': message
            })

    return results


def parse_coverage_xml(xml_path):
    """Parse Coverage XML and extract coverage metrics"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    results = {
        'line_rate': 0.0,
        'branch_rate': 0.0,
        'lines_covered': 0,
        'lines_valid': 0,
        'branches_covered': 0,
        'branches_valid': 0,
        'packages': []
    }

    # Get overall rates
    results['line_rate'] = float(root.get('line-rate', 0))
    results['branch_rate'] = float(root.get('branch-rate', 0))
    results['lines_covered'] = int(root.get('lines-covered', 0))
    results['lines_valid'] = int(root.get('lines-valid', 0))
    results['branches_covered'] = int(root.get('branches-covered', 0))
    results['branches_valid'] = int(root.get('branches-valid', 0))

    # Parse packages/classes
    packages = root.find('packages')
    if packages is not None:
        for package in packages.findall('package'):
            pkg_name = package.get('name', '')
            pkg_line_rate = float(package.get('line-rate', 0))

            classes = []
            classes_elem = package.find('classes')
            if classes_elem is not None:
                for cls in classes_elem.findall('class'):
                    cls_name = cls.get('name', '')
                    cls_filename = cls.get('filename', '')
                    cls_line_rate = float(cls.get('line-rate', 0))

                    classes.append({
                        'name': cls_name,
                        'filename': cls_filename,
                        'line_rate': cls_line_rate
                    })

            results['packages'].append({
                'name': pkg_name,
                'line_rate': pkg_line_rate,
                'classes': classes
            })

    return results


def generate_junit_html(results, output_path):
    """Generate HTML report from JUnit results"""

    pass_rate = (results['passed'] / results['total'] * 100) if results['total'] > 0 else 0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Results Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
        }}
        .header h1 {{ font-size: 28px; margin-bottom: 10px; }}
        .header .subtitle {{ opacity: 0.9; font-size: 14px; }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            padding: 30px;
            background: #f8f9fa;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #667eea;
        }}
        .stat-card.passed {{ border-color: #10b981; }}
        .stat-card.failed {{ border-color: #ef4444; }}
        .stat-card.skipped {{ border-color: #f59e0b; }}
        .stat-label {{
            font-size: 12px;
            text-transform: uppercase;
            color: #6b7280;
            margin-bottom: 8px;
        }}
        .stat-value {{
            font-size: 32px;
            font-weight: bold;
            color: #1f2937;
        }}
        .progress-bar {{
            height: 8px;
            background: #e5e7eb;
            border-radius: 4px;
            overflow: hidden;
            margin: 20px 30px;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #10b981, #059669);
            transition: width 0.3s ease;
        }}
        .tests {{
            padding: 30px;
        }}
        .tests h2 {{
            font-size: 20px;
            margin-bottom: 20px;
            color: #1f2937;
        }}
        .test-item {{
            padding: 15px;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        .test-status {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            flex-shrink: 0;
        }}
        .test-status.passed {{ background: #10b981; }}
        .test-status.failed {{ background: #ef4444; }}
        .test-status.skipped {{ background: #f59e0b; }}
        .test-status.error {{ background: #f97316; }}
        .test-info {{
            flex: 1;
        }}
        .test-name {{
            font-weight: 500;
            color: #1f2937;
            margin-bottom: 4px;
        }}
        .test-class {{
            font-size: 12px;
            color: #6b7280;
        }}
        .test-time {{
            font-size: 12px;
            color: #9ca3af;
            flex-shrink: 0;
        }}
        .test-message {{
            margin-top: 10px;
            padding: 10px;
            background: #fef2f2;
            border-left: 3px solid #ef4444;
            font-size: 13px;
            color: #7f1d1d;
            font-family: 'Courier New', monospace;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Test Results Report</h1>
            <div class="subtitle">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Total Tests</div>
                <div class="stat-value">{results['total']}</div>
            </div>
            <div class="stat-card passed">
                <div class="stat-label">Passed</div>
                <div class="stat-value">{results['passed']}</div>
            </div>
            <div class="stat-card failed">
                <div class="stat-label">Failed</div>
                <div class="stat-value">{results['failed']}</div>
            </div>
            <div class="stat-card skipped">
                <div class="stat-label">Skipped</div>
                <div class="stat-value">{results['skipped']}</div>
            </div>
        </div>

        <div class="progress-bar">
            <div class="progress-fill" style="width: {pass_rate:.1f}%"></div>
        </div>

        <div class="tests">
            <h2>Test Cases ({results['total']})</h2>
"""

    for test in results['testcases']:
        message_html = ''
        if test['message']:
            message_html = f'<div class="test-message">{test["message"]}</div>'

        html += f"""
            <div class="test-item">
                <div class="test-status {test['status']}"></div>
                <div class="test-info">
                    <div class="test-name">{test['name']}</div>
                    <div class="test-class">{test['classname']}</div>
                    {message_html}
                </div>
                <div class="test-time">{test['time']:.3f}s</div>
            </div>
"""

    html += """
        </div>
    </div>
</body>
</html>
"""

    output_path.write_text(html, encoding='utf-8')


def generate_coverage_html(results, output_path):
    """Generate HTML report from Coverage results"""

    line_coverage = results['line_rate'] * 100

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Coverage Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
            color: white;
            padding: 30px;
        }}
        .header h1 {{ font-size: 28px; margin-bottom: 10px; }}
        .header .subtitle {{ opacity: 0.9; font-size: 14px; }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            padding: 30px;
            background: #f8f9fa;
        }}
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #06b6d4;
        }}
        .summary-label {{
            font-size: 12px;
            text-transform: uppercase;
            color: #6b7280;
            margin-bottom: 8px;
        }}
        .summary-value {{
            font-size: 32px;
            font-weight: bold;
            color: #1f2937;
        }}
        .coverage-bar {{
            height: 40px;
            background: #e5e7eb;
            border-radius: 6px;
            overflow: hidden;
            margin: 20px 30px;
            position: relative;
        }}
        .coverage-fill {{
            height: 100%;
            background: linear-gradient(90deg, #10b981, #059669);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 18px;
        }}
        .packages {{
            padding: 30px;
        }}
        .packages h2 {{
            font-size: 20px;
            margin-bottom: 20px;
            color: #1f2937;
        }}
        .package {{
            margin-bottom: 20px;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            overflow: hidden;
        }}
        .package-header {{
            padding: 15px;
            background: #f9fafb;
            font-weight: 500;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .package-name {{ color: #1f2937; }}
        .package-coverage {{
            font-size: 14px;
            color: #6b7280;
        }}
        .class-item {{
            padding: 12px 15px;
            border-top: 1px solid #e5e7eb;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .class-item:hover {{
            background: #f9fafb;
        }}
        .class-name {{
            font-size: 14px;
            color: #4b5563;
        }}
        .class-coverage {{
            font-size: 13px;
            padding: 4px 12px;
            border-radius: 12px;
            font-weight: 500;
        }}
        .coverage-high {{ background: #d1fae5; color: #065f46; }}
        .coverage-medium {{ background: #fef3c7; color: #92400e; }}
        .coverage-low {{ background: #fee2e2; color: #991b1b; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Coverage Report</h1>
            <div class="subtitle">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>

        <div class="summary">
            <div class="summary-card">
                <div class="summary-label">Line Coverage</div>
                <div class="summary-value">{line_coverage:.1f}%</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">Lines Covered</div>
                <div class="summary-value">{results['lines_covered']}</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">Total Lines</div>
                <div class="summary-value">{results['lines_valid']}</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">Branch Coverage</div>
                <div class="summary-value">{results['branch_rate'] * 100:.1f}%</div>
            </div>
        </div>

        <div class="coverage-bar">
            <div class="coverage-fill" style="width: {line_coverage:.1f}%">
                {line_coverage:.1f}% Covered
            </div>
        </div>

        <div class="packages">
            <h2>Packages ({len(results['packages'])})</h2>
"""

    for package in results['packages']:
        pkg_coverage = package['line_rate'] * 100
        html += f"""
            <div class="package">
                <div class="package-header">
                    <span class="package-name">{package['name'] or 'root'}</span>
                    <span class="package-coverage">{pkg_coverage:.1f}% coverage</span>
                </div>
"""

        for cls in package['classes']:
            cls_coverage = cls['line_rate'] * 100

            # Color code based on coverage
            if cls_coverage >= 80:
                coverage_class = 'coverage-high'
            elif cls_coverage >= 50:
                coverage_class = 'coverage-medium'
            else:
                coverage_class = 'coverage-low'

            html += f"""
                <div class="class-item">
                    <span class="class-name">{cls['filename']}</span>
                    <span class="class-coverage {coverage_class}">{cls_coverage:.1f}%</span>
                </div>
"""

        html += """
            </div>
"""

    html += """
        </div>
    </div>
</body>
</html>
"""

    output_path.write_text(html, encoding='utf-8')


def main():
    """Main entry point"""
    results_dir = Path(__file__).parent

    # Parse and convert JUnit XML
    junit_xml = results_dir / 'junit.xml'
    if junit_xml.exists():
        print(f"Converting {junit_xml} to HTML...")
        try:
            junit_results = parse_junit_xml(junit_xml)
            junit_html = results_dir / 'test-report.html'
            generate_junit_html(junit_results, junit_html)
            print(f"✅ Generated: {junit_html}")
        except Exception as e:
            print(f"❌ Error converting JUnit XML: {e}")
    else:
        print(f"⚠️  {junit_xml} not found")

    # Parse and convert Coverage XML
    coverage_xml = results_dir / 'coverage.xml'
    if coverage_xml.exists():
        print(f"Converting {coverage_xml} to HTML...")
        try:
            coverage_results = parse_coverage_xml(coverage_xml)
            coverage_html = results_dir / 'coverage-report.html'
            generate_coverage_html(coverage_results, coverage_html)
            print(f"✅ Generated: {coverage_html}")
        except Exception as e:
            print(f"❌ Error converting Coverage XML: {e}")
    else:
        print(f"⚠️  {coverage_xml} not found")


if __name__ == '__main__':
    main()